"""Audit the complete preregistered control matrix, without reading target data."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median

from microbench.gpu.common.cases import load_cases


def audit(root: Path):
    manifest = json.loads((root / "manifest.json").read_text())
    cases = load_cases("geometry", "control")
    if manifest["splits"] != {"control": cases, "holdout": []}:
        raise ValueError("Run does not match the preregistered control-only matrix")
    groups = {}
    group_features = {}
    for case in cases:
        # Enumerate only declared control paths. Fail closed on absent/rejected
        # rows rather than quietly dropping resource failures or noisy batches.
        row = json.loads((root / "controls" / (case["id"] + ".json")).read_text())
        if (
            row["role"] != "control"
            or row["case"] != case
            or row["cv_group"] != case["cv_group"]
            or row["fingerprint"] != manifest["fingerprint"]
            or row["contaminated"]
        ):
            raise ValueError(f"Invalid control row: {case['id']}")
        latency = float(row["latency_us"])
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError(f"Invalid latency: {case['id']}")
        groups.setdefault(case["cv_group"], {})[
            (case["reuse"], case["num_stages"])
        ] = latency
        group_features.setdefault(case["cv_group"], []).append(row["features"])
    pairs = []
    for group, timings in sorted(groups.items()):
        pairs.append(
            {
                "group": group,
                "aggregate_feature_vectors_identical": all(
                    f == group_features[group][0] for f in group_features[group]
                ),
                "latencies_us": {f"{r}_s{s}": t for (r, s), t in timings.items()},
                "stage2_over_stage1": {
                    r: timings[r, 2] / timings[r, 1] for r in ("none", "a", "ab")
                },
                "reuse_over_disjoint": {
                    f"{r}_s{s}": timings[r, s] / timings["none", s]
                    for r in ("a", "ab")
                    for s in (1, 2)
                },
            }
        )
    return {
        "protocol": "complete control-only matched contrasts; no fit or target reads",
        "fingerprint": manifest["fingerprint"],
        "n": len(cases),
        "groups": len(pairs),
        "median_stage2_over_stage1": {
            r: median(p["stage2_over_stage1"][r] for p in pairs)
            for r in ("none", "a", "ab")
        },
        "median_reuse_over_disjoint": {
            key: median(p["reuse_over_disjoint"][key] for p in pairs)
            for key in ("a_s1", "a_s2", "ab_s1", "ab_s2")
        },
        "pairs": pairs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
