"""Join declared control diagnostics; never fit or admit diagnostic timings."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from triton_viz.performance.gpu_resources import source_resource_features
from triton_viz.tools.gpu_cost_model_pipeline import _write


def audit(root, resource_report):
    manifest = json.loads((root / "manifest.json").read_text())
    resources = json.loads(resource_report.read_text())
    if manifest.get("role") != "control" or resources.get("role") != "control":
        raise ValueError("Only declared control diagnostics are accepted")
    cases = manifest["cases"]
    identifiers = [case["id"] for case in cases]
    resource_ids = [row["case"]["id"] for row in resources["rows"]]
    if len(set(identifiers)) != len(identifiers) or len(set(resource_ids)) != len(
        resource_ids
    ):
        raise ValueError("Duplicate control identity")
    if set(identifiers) != set(resource_ids):
        raise ValueError("Resource and timing control declarations must match")
    by_id = {row["case"]["id"]: row for row in resources["rows"]}
    rows = []
    for case in cases:
        path = root / "controls" / (case["id"] + ".json")
        resource = by_id[case["id"]]
        if resource["case"] != case:
            raise ValueError("Resource control identity mismatch")
        if not path.exists() or resource["status"] != "complete":
            rows.append(dict(case=case, status="missing"))
            continue
        timing = json.loads(path.read_text())
        if timing.get("role") != "control" or timing["case"] != case:
            raise ValueError("Timing control identity mismatch")
        if timing.get("eligible_for_fit") is not False:
            raise ValueError("Expected explicitly diagnostic-only timings")
        samples = [sample["latency_us"] for sample in timing["samples"]]
        if len(samples) != 11 or not all(math.isfinite(x) and x > 0 for x in samples):
            raise ValueError("Incomplete or invalid HES samples")
        median = statistics.median(samples)
        span = (max(samples) - min(samples)) / median
        if (
            timing["median_us"] != median
            or not math.isclose(timing["relative_span"], span)
            or timing["unstable"]
            or span > 0.15
            or timing["graph_kernel_nodes"] != 22
            or timing["dropped_records"] != 0
            or timing["numerical_validation"] != "passed"
        ):
            raise ValueError("Inconsistent or invalid HES diagnostic batch")
        attempt = timing["accepted_attempt"]
        if (
            isinstance(attempt, bool)
            or not isinstance(attempt, int)
            or not 1 <= attempt <= 3
        ):
            raise ValueError("Invalid accepted attempt")
        attempt_path = root / "attempts" / case["id"] / f"{attempt}.json"
        original = json.loads(attempt_path.read_text())
        copied = {
            k: v
            for k, v in timing.items()
            if k not in {"accepted_attempt", "attempt_source"}
        }
        if copied != original:
            raise ValueError("Accepted batch differs from retained attempt")
        features, precision, reasons = source_resource_features(timing["source"])
        rows.append(
            dict(
                case=case,
                status="complete",
                source_features=features,
                source_precision=precision,
                ood_reasons=reasons,
                median_us=median,
                relative_span=span,
                accepted_attempt=attempt,
                registers_per_thread=resource["registers_per_thread"],
                local_bytes_per_thread=resource["local_bytes_per_thread"],
                static_sass_local_counts=resource["static_sass_local_counts"],
            )
        )
    groups = {}
    for row in rows:
        # Pair only controls identical in every declared field except warp count
        # and identifier. This does not identify an isolated spill latency cost.
        key = json.dumps(
            {k: v for k, v in row["case"].items() if k not in {"id", "num_warps"}},
            sort_keys=True,
        )
        groups.setdefault(key, []).append(row)
    pairs = []
    for group in groups.values():
        if len(group) == 2 and all(row["status"] == "complete" for row in group):
            low, high = sorted(group, key=lambda row: row["case"]["num_warps"])
            pairs.append(
                dict(
                    low_warp_control=low["case"]["id"],
                    high_warp_control=high["case"]["id"],
                    latency_ratio_low_over_high=low["median_us"] / high["median_us"],
                    local_bytes_per_thread=[
                        low["local_bytes_per_thread"],
                        high["local_bytes_per_thread"],
                    ],
                )
            )
    return dict(
        role="control",
        eligible_for_fit=False,
        count=len(rows),
        complete=all(row["status"] == "complete" for row in rows),
        rows=rows,
        matched_warp_pairs=pairs,
        caveat="Diagnostic associations, not a fitted source-to-spill or spill-to-latency model. Warp changes also change occupancy and lowering.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Use a new output path")
    result = audit(args.root, args.resources)
    _write(args.output, result)
    print({key: result[key] for key in ("count", "complete", "eligible_for_fit")})


if __name__ == "__main__":
    main()
