"""Collect the fixed 27-control cache counter matrix, never latency samples."""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

from microbench.gpu.common.cache_controls import cache_declaration
from microbench.gpu.harness.measure import snapshot
from triton_viz.tools.gpu_cache_counter_audit import METRICS
from triton_viz.tools.gpu_cost_model_pipeline import _write


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-idle-graphics", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Use a fresh counter root; never overwrite failed controls")
    declaration = cache_declaration("capacity")
    baseline = snapshot(0)
    hardware = {key: baseline[key] for key in ("uuid", "driver", "index")}
    version = subprocess.run(
        [str(args.ncu), "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout
    commands = []
    for mib in declaration["working_set_mib"]:
        for eviction in declaration["evictions"]:
            command = [
                str(args.ncu),
                "--replay-mode",
                "range",
                "--cache-control",
                "all",
                "--clock-control",
                "none",
                "--metrics",
                ",".join(METRICS),
                "--csv",
                "--log-file",
                str(args.output / f"{mib}_{eviction}.csv"),
                sys.executable,
                "-m",
                "triton_viz.tools.gpu_cache_counter_probe",
                "--matrix",
                "capacity",
                "--working-set-mib",
                str(mib),
                "--eviction",
                eviction,
            ]
            if args.allow_idle_graphics:
                command.append("--allow-idle-graphics")
            commands.append(command)
    _write(
        args.output / "manifest.json",
        dict(
            role="control",
            matrix="capacity",
            declaration=declaration,
            hardware=hardware,
            ncu_version=version,
            commands=commands,
            probe_sha256=hashlib.sha256(
                Path(__file__).with_name("gpu_cache_counter_probe.py").read_bytes()
            ).hexdigest(),
            eligible_for_latency_fit=False,
        ),
    )
    for command in commands:
        csv_path = Path(command[command.index("--log-file") + 1])
        with csv_path.with_suffix(".log").open("w") as log:
            try:
                completed = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, timeout=180
                )
            except subprocess.TimeoutExpired:
                log.write(
                    "\nCounter range timed out at 180 seconds; no point removed.\n"
                )
                raise RuntimeError(
                    f"Counter control timed out: {csv_path.stem}"
                ) from None
        if completed.returncode:
            raise RuntimeError(
                f"Counter control failed: {csv_path.stem}; preserve all artifacts"
            )
        print(csv_path.stem, "collected; counter audit still required", flush=True)


if __name__ == "__main__":
    main()
