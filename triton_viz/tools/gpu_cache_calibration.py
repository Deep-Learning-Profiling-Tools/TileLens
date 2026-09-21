"""Control-only effective SDCM calibration with ordinary and nested CV gates.

This calibrates read-miss counts, NOT kernel latency. The calibrated scope is
the declared contiguous-copy range protocol, not arbitrary GPU cache behavior.
Prediction tables depend only on source geometry and candidate coefficients;
measured counters are accessed only by fold-local parameter selection/scoring.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from microbench.gpu.common.cache_controls import cache_declaration
from triton_viz.tools.gpu_cache_counter_model_audit import copy_range_misses
from triton_viz.tools.gpu_cost_model_pipeline import _write


def fit(report):
    declaration = cache_declaration("capacity")
    hardware = report.get("hardware")
    if (
        report.get("role") != "control"
        or report.get("matrix") != "capacity"
        or not report.get("complete")
        or not report.get("replay_counts_consistent")
        or report.get("declaration") != declaration
        or not isinstance(hardware, dict)
        or not hardware.get("uuid")
        or not hardware.get("driver")
    ):
        raise ValueError("Require complete declared capacity-control counters")
    expected = {
        (m, e) for m in declaration["working_set_mib"] for e in declaration["evictions"]
    }
    raw = report["rows"]
    if (
        len(raw) != len(expected)
        or {(r["working_set_mib"], r["eviction"]) for r in raw} != expected
    ):
        raise ValueError("Keep every declared cache control exactly once")
    rows = []
    for r in raw:
        group = f"cache_capacity_mib{r['working_set_mib']}"
        observed = r["counters"]["misses"]
        if (
            r.get("status") != "complete"
            or r.get("cv_group") != group
            or not math.isfinite(observed)
            or observed <= 0
        ):
            raise ValueError("Invalid control identity or miss counter")
        rows.append(
            dict(
                id=f"{r['working_set_mib']}_{r['eviction']}",
                group=group,
                mib=r["working_set_mib"],
                eviction=r["eviction"],
                observed=observed,
            )
        )
    rows.sort(key=lambda r: r["id"])
    groups = sorted({r["group"] for r in rows})
    methods, ways = (
        declaration["probability_methods"],
        declaration["associativity_candidates"],
    )
    predictions = {
        (method, way): [
            copy_range_misses(
                r["mib"] * 1024**2,
                r["eviction"],
                l2_bytes=declaration["l2_bytes"],
                associativity=way,
                tile_bytes=declaration["counter_sector_bytes"],
                method=method,
            )
            for r in rows
        ]
        for method in methods
        for way in ways
    }

    def score(values, indices):
        return sum(
            100 * abs(values[i] / rows[i]["observed"] - 1) for i in indices
        ) / len(indices)

    def train(method, indices):
        return min(ways, key=lambda way: score(predictions[method, way], indices))

    def cv(method, indices):
        values = {}
        for group in sorted({rows[i]["group"] for i in indices}):
            training = [i for i in indices if rows[i]["group"] != group]
            validation = [i for i in indices if rows[i]["group"] == group]
            way = train(method, training)
            values.update({i: predictions[method, way][i] for i in validation})
        return score(values, indices)

    all_indices = list(range(len(rows)))
    ordinary = {method: cv(method, all_indices) for method in methods}
    selected = min(methods, key=ordinary.get)
    final_way = train(selected, all_indices)
    nested_rows = []
    nested_values = {}
    for group in groups:
        training = [i for i in all_indices if rows[i]["group"] != group]
        validation = [i for i in all_indices if rows[i]["group"] == group]
        inner_scores = {method: cv(method, training) for method in methods}
        method = min(methods, key=inner_scores.get)
        way = train(method, training)
        for i in validation:
            prediction = predictions[method, way][i]
            nested_values[i] = prediction
            nested_rows.append(
                dict(
                    id=rows[i]["id"],
                    group=group,
                    training_ids=[rows[j]["id"] for j in training],
                    method=method,
                    associativity=way,
                    predicted_miss_sectors=prediction,
                    observed_miss_sectors=rows[i]["observed"],
                    inner_cv_mape_pct=inner_scores,
                )
            )
    nested_mape = score(nested_values, all_indices)
    passed = ordinary[selected] <= 20 and nested_mape <= 20
    candidate = dict(
        hardware=dict(hardware),
        method=selected,
        effective_associativity=final_way,
        capacity_bytes=declaration["l2_bytes"],
        block_bytes=declaration["counter_sector_bytes"],
    )
    return dict(
        role="control",
        count=len(rows),
        group_count=len(groups),
        ordinary_cv_mape_pct=ordinary,
        nested_cv_mape_pct=nested_mape,
        nested_rows=nested_rows,
        candidate=candidate,
        dual_counter_cv_gate_passed=passed,
        released_counter_model=candidate if passed else None,
        eligible_for_latency_fit=False,
        scope="Three contiguous .cg copy launches, declared capacity range and sweeps; effective uniform-set SDCM, not identified physical cache associativity or a latency gate.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Use a new calibration audit output")
    raw = args.counters.read_bytes()
    result = fit(json.loads(raw))
    result["control_counter_sha256"] = hashlib.sha256(raw).hexdigest()
    _write(args.output, result)
    print(
        {
            k: result[k]
            for k in (
                "ordinary_cv_mape_pct",
                "nested_cv_mape_pct",
                "dual_counter_cv_gate_passed",
            )
        }
    )


if __name__ == "__main__":
    main()
