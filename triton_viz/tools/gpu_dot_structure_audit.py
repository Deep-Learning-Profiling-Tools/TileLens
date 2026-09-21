"""Complete control-only layout contrasts and composition increments."""

import argparse
import json
import math
from pathlib import Path

from microbench.gpu.common.cases import load_cases


def audit(root):
    manifest = json.loads((root / "manifest.json").read_text())
    cases = load_cases("structure", "control")
    if manifest["splits"] != {"control": cases, "holdout": []}:
        raise ValueError("Structure experiment declaration mismatch")
    groups = {}
    for case in cases:
        row = json.loads((root / "controls" / (case["id"] + ".json")).read_text())
        if (
            row["case"] != case
            or row["role"] != "control"
            or row["contaminated"]
            or row["fingerprint"] != manifest["fingerprint"]
            or row["cv_group"] != case["cv_group"]
        ):
            raise ValueError(f"Invalid control provenance: {case['id']}")
        latency = float(row["latency_us"])
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError("Invalid control latency")
        groups.setdefault(case["pair_id"], {})[str(case["variant"])] = latency
    contrasts = []
    for group, times in sorted(groups.items()):
        if "packed" in times:
            value = {"strided_over_packed": times["strided"] / times["packed"]}
        else:
            value = {
                "normalization_increment_us": times["1"] - times["0"],
                "second_dot_increment_us": times["2"] - times["1"],
            }
        contrasts.append({"group": group, "latency_us": times, **value})
    return {
        "n": len(cases),
        "fingerprint": manifest["fingerprint"],
        "protocol": "all declared controls, including negative composition increments",
        "contrasts": contrasts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
