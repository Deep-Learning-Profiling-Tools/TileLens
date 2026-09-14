"""Attribute frozen GPU development-set errors without fitting or retiming."""

import argparse
from collections import defaultdict
import math
from pathlib import Path

from triton_viz.tools.gpu_cost_model_pipeline import _read, _write


def _metrics(rows, total):
    errors = [100 * (r["predicted_us"] / r["measured_us"] - 1) for r in rows]
    absolute = sum(abs(e) for e in errors)
    return {
        "count": len(rows),
        "mape_pct": absolute / len(rows),
        "bias_pct": sum(errors) / len(rows),
        "underprediction_count": sum(e < 0 for e in errors),
        "contribution_pp": absolute / total,
        "ood_count": sum(bool(r["ood_reasons"]) for r in rows),
    }


def audit(root):
    """Read exactly manifest-declared cases; recompute errors from latencies.

    Contributions sum to total MAPE within each disjoint grouping. OOD groups
    overlap and must not be summed or interpreted as causal attribution.
    """
    manifest = _read(root / "manifest.json")
    cases = manifest["cases"]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Expected nonempty, unique manifest cases")
    rows = []
    for case in cases:
        row = _read(root / "cases" / (case["id"] + ".json"))
        if row["case"] != case:
            raise ValueError(f"Case identity mismatch: {case['id']}")
        for key in ("predicted_us", "measured_us"):
            value = row[key]
            if (
                not math.isfinite(value)
                or value < 0
                or (key == "measured_us" and value == 0)
            ):
                raise ValueError(f"Invalid latency: {case['id']}:{key}")
        rows.append(row)
    total = len(rows)
    report = {
        "interpretation": "Development-set diagnostic; no fit, selection, or hardware retiming",
        "calibration_digest": manifest.get("calibration_digest"),
        "overall": _metrics(rows, total),
        "groups": {},
    }
    for fields in (("op",), ("dtype",), ("rows",), ("op", "rows"), ("op", "dtype")):
        groups = defaultdict(list)
        for row in rows:
            key = "/".join(str(row["case"][field]) for field in fields)
            groups[key].append(row)
        report["groups"]["/".join(fields)] = {
            key: _metrics(group, total) for key, group in sorted(groups.items())
        }
    reasons = defaultdict(list)
    for row in rows:
        # Feature values and bounds vary per case; group by reason and feature.
        keys = {":".join(reason.split(":")[:2]) for reason in row["ood_reasons"]}
        for key in keys:
            reasons[key].append(row)
    report["overlapping_ood_groups"] = {
        key: _metrics(group, total) for key, group in sorted(reasons.items())
    }
    report["operator_priority"] = sorted(
        report["groups"]["op"],
        key=lambda op: (-report["groups"]["op"][op]["contribution_pp"], op),
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = audit(args.root)
    if args.output.exists():
        raise ValueError("Use a new audit output path")
    _write(args.output, report)
    print(report["overall"])
    for op in report["operator_priority"]:
        print(op, report["groups"]["op"][op])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
