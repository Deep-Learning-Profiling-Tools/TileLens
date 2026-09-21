"""Collect both dynamic local-traffic counters for all declared pressure controls."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from microbench.gpu.harness.measure import snapshot
from triton_viz.tools.gpu_control_resources import selected_controls
from triton_viz.tools.gpu_cost_model_pipeline import _write

METRICS = (
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
)
CACHE_METRICS = (
    METRICS
    + tuple(
        f"l1tex__t_sectors_pipe_lsu_mem_local_op_{op}_lookup_{outcome}.sum"
        for op in ("ld", "st")
        for outcome in ("hit", "miss")
    )
    + (
        "lts__t_sectors_op_read.sum",
        "lts__t_sectors_op_read_lookup_hit.sum",
        "lts__t_sectors_op_read_lookup_miss.sum",
    )
)
ISSUE_METRICS = (
    "smsp__sass_inst_executed_op_local_ld.sum",
    "smsp__sass_inst_executed_op_local_st.sum",
    "smsp__inst_executed.sum",
    "smsp__inst_issued.sum",
) + tuple(
    f"smsp__warp_issue_stalled_{reason}_per_warp_active.pct"
    for reason in ("long_scoreboard", "short_scoreboard", "barrier", "wait")
)


def reusable_controls(
    root,
    *,
    hardware,
    probe_sha256,
    ncu_version,
    suite="pressure",
    require_monitored=False,
    allowed_graphics=None,
):
    """Validate completed parent rows; preserve failed rows only in their parent."""
    from triton_viz.tools.gpu_local_counter_audit import parse

    manifest = json.loads((root / "manifest.json").read_text())
    if require_monitored and (
        manifest.get("monitored") is not True
        or (
            allowed_graphics is not None
            and manifest.get("allowed_graphics") != list(allowed_graphics)
        )
    ):
        raise ValueError(
            "Cannot inherit unmonitored evidence or changed graphics policy"
        )
    if any(
        manifest.get(k) != v
        for k, v in {
            "role": "control",
            "cases": selected_controls(suite),
            "metrics": list(METRICS),
            "hardware": hardware,
            "probe_sha256": probe_sha256,
            "ncu_version": ncu_version,
        }.items()
    ):
        raise ValueError("Cannot resume a different control protocol or hardware")
    reused = {}
    for case in manifest["cases"]:
        path = root / (case["id"] + ".csv")
        try:
            if manifest.get("monitored") is True:
                from triton_viz.tools.gpu_local_counter_audit import validate_monitor

                validate_monitor(path, manifest)
            text, log = path.read_text(), path.with_suffix(".log").read_text()
            parse(text)
            if (
                f"control={case['id']} numerical=passed profiler_timing_not_for_fit"
                not in log
            ):
                continue
        except (OSError, ValueError):
            continue
        reused[case["id"]] = dict(
            csv_sha256=hashlib.sha256(text.encode()).hexdigest(),
            log_sha256=hashlib.sha256(log.encode()).hexdigest(),
        )
        if manifest.get("monitored") is True:
            reused[case["id"]]["monitor_sha256"] = hashlib.sha256(
                path.with_suffix(".monitor.json").read_bytes()
            ).hexdigest()
    return reused


def commands(
    ncu,
    output,
    *,
    allow_idle_graphics=False,
    suite="pressure",
    cache_lookups=False,
    issue_work=False,
):
    if cache_lookups and issue_work:
        raise ValueError("Declare separate counter phases")
    if suite not in {"pressure", "pressure_pipeline", "resource_dot"}:
        raise ValueError("Require a declared pressure control suite")
    result = []
    for case in selected_controls(suite):
        command = [
            str(ncu),
            "--replay-mode",
            "kernel",
            "--profile-from-start",
            "off",
            "--cache-control",
            "all",
            "--clock-control",
            "none",
            "--metrics",
            ",".join(
                ISSUE_METRICS
                if issue_work
                else CACHE_METRICS
                if cache_lookups
                else METRICS
            ),
            "--csv",
            "--log-file",
            str(output / (case["id"] + ".csv")),
            sys.executable,
            "-m",
            "triton_viz.tools.gpu_control_counter_probe",
            "--suite",
            suite,
            "--case-id",
            case["id"],
        ]
        if allow_idle_graphics:
            command.append("--allow-idle-graphics")
        result.append(command)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("pressure", "pressure_pipeline", "resource_dot"),
        default="pressure",
    )
    parser.add_argument("--ncu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-idle-graphics", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--monitor", action="store_true")
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--cache-lookups", action="store_true")
    phase.add_argument("--issue-work", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Require a fresh control output directory")
    if (args.cache_lookups or args.issue_work) and args.resume_from is not None:
        raise ValueError("Additional counter phases require fresh complete collection")
    if (
        args.monitor
        and args.resume_from is not None
        and json.loads((args.resume_from / "manifest.json").read_text()).get(
            "monitored"
        )
        is not True
    ):
        raise ValueError("Monitored collection cannot inherit unmonitored evidence")
    baseline = snapshot(0)
    version = subprocess.run(
        [str(args.ncu), "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout
    declared = commands(
        args.ncu,
        args.output,
        allow_idle_graphics=args.allow_idle_graphics,
        suite=args.suite,
        cache_lookups=args.cache_lookups,
        issue_work=args.issue_work,
    )
    hardware = {k: baseline[k] for k in ("uuid", "driver", "index")}
    probe_sha256 = hashlib.sha256(
        Path(__file__).with_name("gpu_control_counter_probe.py").read_bytes()
    ).hexdigest()
    reused = (
        {}
        if args.resume_from is None
        else reusable_controls(
            args.resume_from,
            hardware=hardware,
            probe_sha256=probe_sha256,
            ncu_version=version,
            suite=args.suite,
            require_monitored=args.monitor,
            allowed_graphics=[p["pid"] for p in baseline["graphics_processes"]]
            if args.allow_idle_graphics
            else [],
        )
    )
    _write(
        args.output / "manifest.json",
        dict(
            role="control",
            cases=selected_controls(args.suite),
            suite=args.suite,
            commands=declared,
            monitored=args.monitor,
            allowed_graphics=[p["pid"] for p in baseline["graphics_processes"]]
            if args.allow_idle_graphics
            else [],
            hardware=hardware,
            ncu_version=version,
            metrics=list(
                ISSUE_METRICS
                if args.issue_work
                else CACHE_METRICS
                if args.cache_lookups
                else METRICS
            ),
            cache_lookup_phase=args.cache_lookups,
            issue_work_phase=args.issue_work,
            eligible_for_latency_fit=False,
            probe_sha256=probe_sha256,
            inherited_controls=reused,
            parent=None
            if args.resume_from is None
            else dict(
                root=str(args.resume_from.resolve()),
                manifest_sha256=hashlib.sha256(
                    (args.resume_from / "manifest.json").read_bytes()
                ).hexdigest(),
            ),
        ),
    )
    for command in declared:
        path = Path(command[command.index("--log-file") + 1])
        if path.stem in reused:
            for suffix in (
                (".csv", ".log", ".monitor.json") if args.monitor else (".csv", ".log")
            ):
                original = args.resume_from / (path.stem + suffix)
                content = original.read_bytes()
                if (
                    hashlib.sha256(content).hexdigest()
                    != reused[path.stem][
                        ("monitor" if suffix == ".monitor.json" else suffix[1:])
                        + "_sha256"
                    ]
                ):
                    raise ValueError("Parent control changed during resume")
                path.with_suffix(suffix).write_bytes(content)
            print(path.stem, "inherited unchanged from verified parent", flush=True)
            continue
        if args.monitor:
            from triton_viz.tools.gpu_cupti_perturbation_collect import (
                monitored_process,
            )

            monitoring = monitored_process(
                command,
                path.with_suffix(".log"),
                allowed_graphics=tuple(p["pid"] for p in baseline["graphics_processes"])
                if args.allow_idle_graphics
                else (),
                timeout=180,
                own_process_group=True,
            )
            _write(
                path.with_suffix(".monitor.json"),
                dict(
                    case_id=path.stem,
                    monitoring=monitoring,
                    log_sha256=hashlib.sha256(
                        path.with_suffix(".log").read_bytes()
                    ).hexdigest(),
                    csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
                    if path.exists()
                    else None,
                ),
            )
            if monitoring["returncode"] or monitoring["contaminated"]:
                raise RuntimeError(
                    f"Monitored counter failure: {path.stem}; preserve all artifacts"
                )
            print(path.stem, "collected with process-group monitoring", flush=True)
            continue
        with path.with_suffix(".log").open("w") as log:
            try:
                completed = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, timeout=180
                )
            except subprocess.TimeoutExpired:
                log.write("\nTimed out at 180 seconds; no control removed.\n")
                raise RuntimeError(f"Counter timeout: {path.stem}") from None
        if completed.returncode:
            raise RuntimeError(f"Counter failure: {path.stem}; preserve all artifacts")
        print(path.stem, "collected; counter validation still required", flush=True)


if __name__ == "__main__":
    main()
