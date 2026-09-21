"""Collect both dynamic local-traffic counters for all declared pressure controls."""

import argparse
import hashlib
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
    _write(
        args.output / "manifest.json",
        dict(
            role="control",
            cases=selected_controls("pressure"),
            commands=declared,
            hardware={k: baseline[k] for k in ("uuid", "driver", "index")},
            ncu_version=version,
            metrics=list(METRICS),
            eligible_for_latency_fit=False,
            probe_sha256=hashlib.sha256(
                Path(__file__).with_name("gpu_control_counter_probe.py").read_bytes()
            ).hexdigest(),
        ),
    )
    for command in declared:
        path = Path(command[command.index("--log-file") + 1])
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
