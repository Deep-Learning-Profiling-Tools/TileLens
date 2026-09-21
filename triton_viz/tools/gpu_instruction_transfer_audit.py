"""Control-only transfer of instruction work, not compiler-at-prediction pricing.

Measured warp instruction totals are normalized by source program/warp counts.
The source query contains no measured instruction, cache, resource or time label.
Single/repeated dot support remains explicit; generic composed regions need
their own lowering mapping rather than invented zero instruction overhead.
"""

import math

from triton_viz.tools.gpu_spill_transfer_audit import (
    FEATURES,
    dot_instructions,
    request_candidates,
)

LABELS = ("LDL", "STL", "executed", "issued")


def scale(row):
    value = dot_instructions(row) * row["source_features"]["dots_per_program"]
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Invalid source dot instruction scale")
    return value


def predict(training, source):
    if not training or any(r.get("role") != "control" for r in training):
        raise ValueError("Require control instruction training")
    for row in [*training, source]:
        if any(
            not math.isfinite(row["source_features"][k])
            or row["source_features"][k] < 0
            for k in FEATURES
        ):
            raise ValueError("Invalid source descriptor")
    for row in training:
        if any(
            not math.isfinite(row["instructions_per_warp"][k])
            or row["instructions_per_warp"][k] < 0
            for k in LABELS
        ):
            raise ValueError("Invalid instruction label")
    domain = {
        k: (
            min(r["source_features"][k] for r in training),
            max(r["source_features"][k] for r in training),
        )
        for k in FEATURES
    }
    candidates, reasons = request_candidates(training, source, separate_single_dot=True)
    distances = [
        sum(
            abs(source["source_features"][k] - r["source_features"][k]) / (hi - lo)
            if hi > lo
            else float(source["source_features"][k] != lo)
            for k, (lo, hi) in domain.items()
        )
        for r in candidates
    ]
    nearest = [r for r, d in zip(candidates, distances) if d == min(distances)]
    predicted = {
        k: scale(source)
        * sum(r["instructions_per_warp"][k] / scale(r) for r in nearest)
        / len(nearest)
        for k in LABELS
    }
    return dict(
        instructions_per_warp=predicted,
        neighbors=[r["case"]["id"] for r in nearest],
        ood_reasons=reasons
        + [
            f"instruction_domain:{k}"
            for k, (lo, hi) in domain.items()
            if not lo <= source["source_features"][k] <= hi
        ],
    )


def validate(rows):
    if not rows or any(r.get("role") != "control" for r in rows):
        raise ValueError("Require control instruction rows")
    if len({r["case"]["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate controls")
    owners = {}
    for row in rows:
        for a, b in row["dot_shapes"]:
            key = (tuple(a), tuple(b))
            group = row["case"]["cv_group"]
            if key in owners and owners[key] != group:
                raise ValueError("Shared geometry crosses validation partitions")
            owners[key] = group
        if (
            isinstance(row["source_program_count"], bool)
            or not isinstance(row["source_program_count"], int)
            or row["source_program_count"] < 1
        ):
            raise ValueError("Invalid source program count")
    groups = sorted({r["case"]["cv_group"] for r in rows})
    if len(groups) < 3:
        raise ValueError("Need three geometry groups")
    predictions = []
    for group in groups:
        training = [r for r in rows if r["case"]["cv_group"] != group]
        for row in rows:
            if row["case"]["cv_group"] != group:
                continue
            predictions.append(
                dict(
                    case=row["case"],
                    actual=row["instructions_per_warp"],
                    warp_count=row["source_program_count"]
                    * row["source_features"]["threads_per_program"]
                    / 32,
                    training_ids=[r["case"]["id"] for r in training],
                    **predict(training, row),
                )
            )
    metrics = {}
    for name in LABELS:
        measured = sum(r["actual"][name] * r["warp_count"] for r in predictions)
        difference = sum(
            abs(r["actual"][name] - r["instructions_per_warp"][name]) * r["warp_count"]
            for r in predictions
        )
        metrics[name] = dict(
            instruction_count_wape_pct=100 * difference / measured
            if measured
            else None,
            mae_instructions_per_warp=sum(
                abs(r["actual"][name] - r["instructions_per_warp"][name])
                for r in predictions
            )
            / len(predictions),
            false_positive=sum(
                r["actual"][name] == 0 and r["instructions_per_warp"][name] > 0
                for r in predictions
            ),
            false_negative=sum(
                r["actual"][name] > 0 and r["instructions_per_warp"][name] == 0
                for r in predictions
            ),
        )
    return dict(
        role="control",
        count=len(rows),
        eligible_for_fit=False,
        released_model=None,
        rows=predictions,
        metrics=metrics,
        ood_count=sum(bool(r["ood_reasons"]) for r in predictions),
        caveat="Instruction-work transfer only, not cache or latency validation. All controls retained; no target read.",
    )
