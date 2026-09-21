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


def reusable_controls(root, *, hardware, probe_sha256, ncu_version):
    """Validate completed parent rows; preserve failed rows only in their parent."""
    from triton_viz.tools.gpu_local_counter_audit import parse

    manifest = json.loads((root / "manifest.json").read_text())
    if any(
        manifest.get(k) != v
        for k, v in {
            "role": "control",
            "cases": selected_controls("pressure"),
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
    return reused


def commands(ncu, output, *, allow_idle_graphics=False):
    result = []
    for case in selected_controls("pressure"):
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
            ",".join(METRICS),
            "--csv",
            "--log-file",
            str(output / (case["id"] + ".csv")),
            sys.executable,
            "-m",
            "triton_viz.tools.gpu_control_counter_probe",
            "--suite",
            "pressure",
            "--case-id",
            case["id"],
        ]
        if allow_idle_graphics:
            command.append("--allow-idle-graphics")
        result.append(command)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-idle-graphics", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Require a fresh control output directory")
    baseline = snapshot(0)
    version = subprocess.run(
        [str(args.ncu), "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout
    declared = commands(
        args.ncu, args.output, allow_idle_graphics=args.allow_idle_graphics
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
        )
    )
    _write(
        args.output / "manifest.json",
        dict(
            role="control",
            cases=selected_controls("pressure"),
            commands=declared,
            hardware=hardware,
            ncu_version=version,
            metrics=list(METRICS),
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
            for suffix in (".csv", ".log"):
                original = args.resume_from / (path.stem + suffix)
                content = original.read_bytes()
                if (
                    hashlib.sha256(content).hexdigest()
                    != reused[path.stem][suffix[1:] + "_sha256"]
                ):
                    raise ValueError("Parent control changed during resume")
                path.with_suffix(suffix).write_bytes(content)
            print(path.stem, "inherited unchanged from verified parent", flush=True)
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
