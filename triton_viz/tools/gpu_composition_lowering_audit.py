"""Matched control compiler diagnostics for single-dot versus composition lowering.

No latency reads, model fitting, or prediction-time compiler access. Every input
control stays accounted for; geometry controls without composition pairing are
listed separately, not silently deleted from a calibration dataset.
"""

import argparse
import json
from pathlib import Path

from triton_viz.tools.gpu_control_resource_audit import audit as resource_audit
from triton_viz.tools.gpu_cost_model_pipeline import _write


KEYS = ("bm", "bn", "dtype", "precision", "programs", "num_warps", "num_stages")


def audit(parents, components):
    for report in (parents, components):
        if report.get("role") != "control" or not report.get("complete"):
            raise ValueError("Require complete control-only resource audits")
    rows = parents["rows"] + components["rows"]
    if not components["rows"]:
        raise ValueError("Require declared component controls")
    ids = [r["case"]["id"] for r in rows]
    if len(ids) != len(set(ids)) or any(r["status"] != "complete" for r in rows):
        raise ValueError("Duplicate or incomplete controls")
    groups, unpaired = {}, []
    for row in parents["rows"]:
        case = row["case"]
        if case["kind"] != "structure_composition":
            unpaired.append(case["id"])
            continue
        key = tuple(case[k] for k in KEYS)
        group = groups.setdefault(key, {})
        if case["variant"] in group:
            raise ValueError("Duplicate composition phase")
        group[case["variant"]] = row
    comparisons = []
    seen = set()
    for component in components["rows"]:
        case = component["case"]
        key = tuple(case[k] for k in KEYS)
        if key in seen or key not in groups or set(groups[key]) != {0, 1, 2}:
            raise ValueError("Missing or duplicate matched phases")
        seen.add(key)
        phases = groups[key]
        if (
            case["kind"] != "geometry_dot"
            or case["bk"] != case["bn"]
            or case["repeat"] != 1
            or case["reuse"] != "none"
            or any(p["case"]["cv_group"] != case["cv_group"] for p in phases.values())
            or any(
                p["case"]["bk"] != phases[0]["case"]["bk"] or p["case"]["repeat"] != 1
                for p in phases.values()
            )
        ):
            raise ValueError("Component geometry or CV pairing mismatch")

        def resources(row):
            return {
                k: row[k]
                for k in (
                    "registers_per_thread",
                    "local_bytes_per_thread",
                    "shared_bytes",
                )
            }

        comparisons.append(
            dict(
                component_id=case["id"],
                cv_group=case["cv_group"],
                parent_ids=[phases[i]["case"]["id"] for i in range(3)],
                first_dot=resources(phases[0]),
                first_dot_normalized=resources(phases[1]),
                second_dot_alone=resources(component),
                composition=resources(phases[2]),
                local_allocation_only_in_composition=(
                    phases[2]["local_bytes_per_thread"] > 0
                    and phases[1]["local_bytes_per_thread"] == 0
                    and component["local_bytes_per_thread"] == 0
                ),
            )
        )
    if seen != set(groups):
        raise ValueError("Some composition controls lack a declared component")
    return dict(
        role="control",
        eligible_for_fit=False,
        input_control_count=len(rows),
        paired_control_count=4 * len(comparisons),
        unpaired_parent_ids=unpaired,
        rows=comparisons,
        caveat="Static allocation comparisons isolate lowering differences, not dynamic spill traffic or additive latency costs.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parents", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Use a new audit output")
    result = audit(resource_audit(args.parents), resource_audit(args.components))
    _write(args.output, result)
    print(
        json.dumps(
            dict(
                comparisons=len(result["rows"]),
                composition_only_local_allocations=sum(
                    r["local_allocation_only_in_composition"] for r in result["rows"]
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
