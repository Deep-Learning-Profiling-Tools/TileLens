"""Run all declared pressure controls through diagnostic-only cold HES probes.

Use at most three whole-batch attempts per control, retaining logs and samples.
Stop if none passes; accept the first valid batch, never the fastest. This is not
a fitter and does not authorize mixing these samples with old warm-cache data.
"""

import argparse
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
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Use a fresh diagnostic root")
    cases = load_cases("pressure", "control")
    _write(
        args.output / "manifest.json",
        dict(
            role="control",
            cases=cases,
            eligible_for_fit=False,
            attempt_policy="first complete stable batch, maximum 3; retain every attempt",
            metric="cupti_hes_kernel_us_eviction_unvalidated",
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
            "pressure",
            "--case-id",
            case["id"],
            "--graph-samples",
        ]
        if args.allow_idle_graphics:
            command.append("--allow-idle-graphics")
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
            if row.get("failed") or row["unstable"]:
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
