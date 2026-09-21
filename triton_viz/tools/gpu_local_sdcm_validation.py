"""Control-only grouped validation of the cyclic local SDCM hypothesis.

This does not publish coefficients or apply a latency gate. Zero observed load
misses are retained: selection uses load hit-fraction MAE, not epsilon MAPE.
"""

import argparse
import json
import math
from pathlib import Path

from triton_viz.performance.gpu_cache import cyclic_local_cache_hypothesis
from triton_viz.tools.gpu_cost_model_pipeline import _write
from triton_viz.tools.gpu_cta_schedule_audit import audit_schedule
from triton_viz.tools.gpu_local_cache_counter_audit import audit_grid


def scenarios():
    # Broad power-of-two effective-cache hypotheses, not physical capacity or
    # associativity claims. Frozen independently of target/holdout performance.
    return [
        dict(capacity_bytes=1024 * kib, associativity=ways)
        for kib in (16, 32, 64, 128, 256, 512, 1024)
        for ways in (1, 2, 4, 8, 16, 32, 64)
    ]


def prediction(row, scenario):
    programs, sms = row["programs"], row["sm_count"]
    if programs < sms or programs % sms:
        raise ValueError("Uniform whole-CTA per-SM scenario requires divisible grid")
    # Four warps per CTA, four 32-byte sectors per warp's FP32 slot.
    result = cyclic_local_cache_hypothesis(
        actors=4 * (programs // sms),
        slots=row["local_slots"],
        blocks_per_slot=4,
        iterations=65536,
        capacity_blocks=scenario["capacity_bytes"] // 32,
        associativity=scenario["associativity"],
        method="exact",
    )["traffic"]
    return {
        op: result[f"{op}_hits"] / result[f"{op}_requests"] for op in ("load", "store")
    }


def validate(rows):
    if len(rows) < 3:
        raise ValueError("Need enough control groups for nested validation")
    for row in rows:
        if any(
            not math.isfinite(row[op]) or not 0 <= row[op] <= 1
            for op in ("load", "store")
        ):
            raise ValueError("Invalid observed hit fraction")
    candidates = scenarios()
    predictions = [
        [prediction(row, candidate) for row in rows] for candidate in candidates
    ]

    def select(indices):
        # This is the ONLY label-dependent parameter selection; test fold
        # labels never appear in indices. Deterministic ties follow declaration.
        return min(
            range(len(candidates)),
            key=lambda c: sum(
                abs(predictions[c][i]["load"] - rows[i]["load"]) for i in indices
            ),
        )

    def score(pairs):
        result = {}
        for op in ("load", "store"):
            observed_misses = sum(
                rows[i]["programs"]
                * 16
                * (65536 + rows[i]["local_slots"])
                * (1 - rows[i][op])
                for i, _ in pairs
            )
            miss_error = sum(
                rows[i]["programs"]
                * 16
                * (65536 + rows[i]["local_slots"])
                * abs(predictions[c][i][op] - rows[i][op])
                for i, c in pairs
            )
            result[op] = dict(
                mae_percentage_points=100
                * sum(abs(predictions[c][i][op] - rows[i][op]) for i, c in pairs)
                / len(pairs),
                max_error_percentage_points=100
                * max(abs(predictions[c][i][op] - rows[i][op]) for i, c in pairs),
                observed_miss_sectors=observed_misses,
                absolute_miss_sector_error=miss_error,
                miss_count_wape_pct=100 * miss_error / observed_misses
                if observed_misses
                else None,
            )
        return result

    axes = {}
    for axis in ("footprint_bytes", "local_slots", "programs"):
        groups = sorted({r[axis] for r in rows})
        if len(groups) < 3:
            raise ValueError("Nested CV requires at least three groups per axis")
        outer_pairs, folds = [], []
        for group in groups:
            train = [i for i, r in enumerate(rows) if r[axis] != group]
            test = [i for i, r in enumerate(rows) if r[axis] == group]
            selected = select(train)
            inner_pairs, inner = [], []
            for held in groups:
                if held == group:
                    continue
                fit_indices = [i for i in train if rows[i][axis] != held]
                validation = [i for i in train if rows[i][axis] == held]
                candidate = select(fit_indices)
                inner_pairs.extend((i, candidate) for i in validation)
                inner.append(dict(held_group=held, selected=candidates[candidate]))
            pairs = [(i, selected) for i in test]
            outer_pairs.extend(pairs)
            folds.append(
                dict(
                    held_group=group,
                    selected=candidates[selected],
                    outer_score=score(pairs),
                    inner_score=score(inner_pairs),
                    inner_folds=inner,
                    predictions=[
                        dict(
                            programs=rows[i]["programs"],
                            local_slots=rows[i]["local_slots"],
                            predicted=predictions[selected][i],
                            observed={op: rows[i][op] for op in ("load", "store")},
                        )
                        for i in test
                    ],
                )
            )
        axes[axis] = dict(outer_score=score(outer_pairs), folds=folds)
    final = select(list(range(len(rows))))
    return dict(
        role="control",
        eligible_for_fit=False,
        count=len(rows),
        scenarios=candidates,
        diagnostic_full_control_scenario=candidates[final],
        full_control_score=score([(i, final) for i in range(len(rows))]),
        grouped_validation=axes,
        released_model=None,
        caveat="Hit-fraction errors, NOT miss-count MAPE or latency CV. Inner folds diagnose parameter stability; all outer labels withheld from selection. No coefficients admitted to prediction.",
    )


def run(root):
    root = Path(root)
    report = audit_grid(root)
    schedule = audit_schedule(root)
    rows = []
    for row, placement in zip(report["rows"], schedule["rows"]):
        p, s = row["programs"], row["local_slots"]
        native = json.loads((root / f"local_s{s}_p{p}_i65536.json").read_text())[
            "native_audit"
        ]
        sms = int(native["metadata"]["sm_count"])
        if (
            not placement["sm_observed"]
            or placement["changed_endpoint_ctas"]
            or len(placement["start_sm_counts"]) != sms
            or set(placement["start_sm_counts"].values()) != {p // sms}
        ):
            raise ValueError(
                "Uniform SM assignment hypothesis not validated; retain raw controls"
            )
        lookup = row.get("local_lookups")
        if lookup is None or any(
            v["replay_count_disagreement"] != 0 for v in lookup.values()
        ):
            raise ValueError("Require complete consistent direct local lookup evidence")
        rows.append(
            dict(
                programs=p,
                local_slots=s,
                sm_count=sms,
                footprint_bytes=row["footprint_bytes"],
                **{op: lookup[op]["hit_fraction"] for op in ("load", "store")},
            )
        )
    return validate(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Require fresh output")
    result = run(args.root)
    _write(args.output, result)
    print(
        json.dumps(
            {
                axis: value["outer_score"]
                for axis, value in result["grouped_validation"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
