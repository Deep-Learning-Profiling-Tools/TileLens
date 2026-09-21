"""Run all declared pressure controls through diagnostic-only cold HES probes.

Use at most three whole-batch attempts per control, retaining logs and samples.
Stop if none passes; accept the first valid batch, never the fastest. This is not
a fitter and does not authorize mixing these samples with old warm-cache data.
"""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

from microbench.gpu.common.cases import load_cases
from triton_viz.tools.gpu_cost_model_pipeline import _write


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-idle-graphics", action="store_true")
    parser.add_argument(
        "--suite",
        choices=(
            "pressure",
            "pressure_pipeline",
            "resource_transfer",
            "composition_component",
            "geometry",
            "structure",
            "stability",
            "coverage",
        ),
        default="pressure",
    )
    parser.add_argument("--monitored", action="store_true")
    parser.add_argument("--capture-cache", action="store_true")
    parser.add_argument("--kernels-per-sample", type=int, default=1)
    parser.add_argument(
        "--timestamp-method", choices=("hes", "software_serial"), default="hes"
    )
    parser.add_argument(
        "--launch-mode", choices=("graph", "individual"), default="graph"
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Use a fresh diagnostic root")
    if args.kernels_per_sample < 1:
        raise ValueError("A positive sample replication count is required")
    cases = load_cases(args.suite, "control")
    _write(
        args.output / "manifest.json",
        dict(
            role="control",
            suite=args.suite,
            monitored=args.monitored,
            capture_cache=args.capture_cache,
            kernels_per_sample=args.kernels_per_sample,
            timestamp_method=args.timestamp_method,
            launch_mode=args.launch_mode,
            packages={
                name: importlib.metadata.version(name) for name in ("torch", "triton")
            },
            collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            probe_sha256=hashlib.sha256(
                Path(__file__).with_name("gpu_cupti_probe.py").read_bytes()
            ).hexdigest(),
            library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest()
            if args.library.is_file()
            else None,
            cases=cases,
            eligible_for_fit=False,
            attempt_policy="first complete stable batch, maximum 3; retain every attempt",
            metric="cupti_software_serial_group_mean_kernel_us_eviction_unvalidated"
            if args.timestamp_method == "software_serial"
            else "cupti_hes_kernel_us_eviction_unvalidated"
            if args.kernels_per_sample == 1
            else "cupti_hes_group_mean_kernel_us_eviction_unvalidated",
        ),
    )
    for case in cases:
        command = [
            sys.executable,
            "-u",
            "-m",
            "triton_viz.tools.gpu_cupti_probe",
            "--library",
            str(args.library),
            "--output",
            "",
            "--suite",
            args.suite,
            "--case-id",
            case["id"],
            "--timestamp-method",
            args.timestamp_method,
            "--kernels-per-sample",
            str(args.kernels_per_sample),
        ]
        if args.launch_mode == "graph":
            command.append("--graph-samples")
        if args.allow_idle_graphics:
            command.append("--allow-idle-graphics")
        if args.monitored:
            command.append("--monitored")
        if args.capture_cache:
            command.append("--capture-cache")
        accepted = None
        for attempt in range(1, 4):
            directory = args.output / "attempts" / case["id"]
            directory.mkdir(parents=True, exist_ok=True)
            output = directory / f"{attempt}.json"
            command[command.index("--output") + 1] = str(output)
            with output.with_suffix(".log").open("w") as log:
                try:
                    result = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, timeout=180
                    )
                except subprocess.TimeoutExpired:
                    log.write("\nDiagnostic subprocess timed out after 180 seconds.\n")
                    print(case["id"], "timed out attempt", attempt, flush=True)
                    continue
            if result.returncode:
                # A launch/instrumentation failure is not evidence of a slow
                # kernel. Preserve the failed attempt and retry from a new process.
                print(case["id"], "failed attempt", attempt, flush=True)
                continue
            row = json.loads(output.read_text())
            if (
                row.get("failed")
                or row["unstable"]
                or (args.monitored and row.get("contaminated") is not False)
            ):
                print(case["id"], "unstable/invalid attempt", attempt, flush=True)
                continue
            accepted = {
                **row,
                "accepted_attempt": attempt,
                "attempt_source": str(output),
            }
            break
        if accepted is None:
            raise RuntimeError(
                f"Control probe failed/unstable after 3 batches: {case['id']}; all attempts retained"
            )
        _write(args.output / "controls" / (case["id"] + ".json"), accepted)
        print(case["id"], accepted["median_us"], accepted["relative_span"], flush=True)


if __name__ == "__main__":
    main()
