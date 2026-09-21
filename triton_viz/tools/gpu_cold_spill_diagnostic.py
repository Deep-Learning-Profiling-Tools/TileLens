"""Counter-trained request service pricing; diagnostic only, never admission.

Pure-dot controls use source-inferred instruction-normalized requests. Composed
controls remain in every score using the source baseline with an explicit
unsupported-spill-mapping flag, not an invented zero-traffic prediction.
"""

import math

from triton_viz.tools.gpu_cold_resource_diagnostic import (
    SERVICE_FEATURES,
    service_model,
    service_prediction,
)
from triton_viz.tools.gpu_spill_transfer_audit import (
    FEATURES,
    OPS,
    dot_instructions,
    request_candidates,
)
from triton_viz.tools.gpu_instruction_transfer_audit import (
    WORK_LABELS,
    decompose_work,
    predict_work,
)

REQUESTS = tuple("predicted_wave_" + op for op in OPS)
INSTRUCTION_WORK = tuple("predicted_wave_instruction_" + op for op in WORK_LABELS)


def request_prediction(training, source, *, separate_single_dot=False):
    if not training or any(r.get("role") != "control" for r in training):
        raise ValueError("Require nonempty control counter training")
    domain = {
        k: (
            min(r["source_features"][k] for r in training),
            max(r["source_features"][k] for r in training),
        )
        for k in FEATURES
    }
    candidates, regime_reasons = request_candidates(
        training, source, separate_single_dot=separate_single_dot
    )
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

    def scale(row):
        value = row["source_features"]["dots_per_program"] * dot_instructions(row)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid source instruction scale")
        return value

    predicted = {
        op: scale(source)
        * sum(r["counter_bytes_per_thread"][op] / scale(r) for r in nearest)
        / len(nearest)
        for op in OPS
    }
    return dict(
        bytes_per_thread=predicted,
        ood_reasons=regime_reasons
        + [
            f"counter_domain:{k}"
            for k, (lo, hi) in domain.items()
            if not lo <= source["source_features"][k] <= hi
        ],
    )


def validate(
    counter_rows, latency_rows, *, include_regime=False, instruction_rows=None
):
    shape_groups = {}
    for rows in (counter_rows, latency_rows) + (
        (instruction_rows,) if instruction_rows is not None else ()
    ):
        if not rows or any(r.get("role") != "control" for r in rows):
            raise ValueError("Require control-only data")
        if len({r["case"]["id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate controls")
        for row in rows:
            for a, b in row["dot_shapes"]:
                key = (tuple(a), tuple(b))
                group = row["case"]["cv_group"]
                if key in shape_groups and shape_groups[key] != group:
                    raise ValueError("Shared source geometry crosses partitions")
                shape_groups[key] = group
            if any(
                not math.isfinite(row["source_features"][k])
                or row["source_features"][k] < 0
                for k in FEATURES
            ):
                raise ValueError("Invalid source descriptor")
    for row in counter_rows:
        if any(
            not math.isfinite(row["counter_bytes_per_thread"][op])
            or row["counter_bytes_per_thread"][op] < 0
            for op in OPS
        ):
            raise ValueError("Invalid measured counter")
    if instruction_rows is not None:
        for row in instruction_rows:
            decompose_work(row)
    for row in latency_rows:
        if (
            row.get("eligible_for_fit") is not False
            or not math.isfinite(row["latency_us"])
            or row["latency_us"] <= 0
        ):
            raise ValueError("Require explicit diagnostic timing rows")
        if any(
            not math.isfinite(row["pricing_features"][k])
            or row["pricing_features"][k] < 0
            for k in (*SERVICE_FEATURES, "waves")
        ):
            raise ValueError("Invalid service work")
    groups = sorted({r["case"]["cv_group"] for r in latency_rows})
    if len(groups) < 3:
        raise ValueError("Need three geometry groups")
    names = ("source", "source_pure", "source_plus_requests")
    if include_regime:
        names += ("source_plus_regime_requests",)
    if instruction_rows is not None:
        names += ("source_plus_instruction_work",)
    cache = {}

    def fold(excluded):
        key = tuple(sorted(excluded))
        if key in cache:
            return cache[key]
        counters = [r for r in counter_rows if r["case"]["cv_group"] not in excluded]
        instructions = (
            [r for r in instruction_rows if r["case"]["cv_group"] not in excluded]
            if instruction_rows is not None
            else None
        )
        projected = []
        for row in latency_rows:
            pure = row["case"]["kind"] == "geometry_dot"
            result = request_prediction(counters, row) if pure else None
            regime = (
                request_prediction(counters, row, separate_single_dot=True)
                if pure and include_regime
                else None
            )
            features = {k: row["pricing_features"][k] for k in SERVICE_FEATURES}
            regime_features = features.copy()
            instruction_features = features.copy()
            work = (
                predict_work(instructions, row)
                if pure and instructions is not None
                else None
            )
            if work is not None:
                for op, k in zip(WORK_LABELS, INSTRUCTION_WORK):
                    instruction_features[k] = (
                        work["instructions_per_warp"][op]
                        * row["source_features"]["threads_per_program"]
                        / 32
                        * row["pricing_features"]["waves"]
                    )
            if pure:
                for op, k in zip(OPS, REQUESTS):
                    features[k] = (
                        result["bytes_per_thread"][op]
                        * row["source_features"]["threads_per_program"]
                        / 32
                        * row["pricing_features"]["waves"]
                    )
                    if regime is not None:
                        regime_features[k] = (
                            regime["bytes_per_thread"][op]
                            * row["source_features"]["threads_per_program"]
                            / 32
                            * row["pricing_features"]["waves"]
                        )
            projected.append(
                dict(
                    id=row["case"]["id"],
                    group=row["case"]["cv_group"],
                    pure=pure,
                    features=features,
                    regime_features=regime_features,
                    instruction_features=instruction_features,
                    instruction_ood=work["ood_reasons"]
                    if work is not None
                    else ["instruction_mapping_unmodeled_composition"],
                    regime_ood=regime["ood_reasons"]
                    if regime
                    else ["spill_mapping_unmodeled_composition"],
                    latency_us=row["latency_us"],
                    counter_ood=result["ood_reasons"]
                    if pure
                    else ["spill_mapping_unmodeled_composition"],
                )
            )
        training = [r for r in projected if r["group"] not in excluded]
        pure_training = [r for r in training if r["pure"]]
        models = {
            "source": service_model(training, SERVICE_FEATURES),
            "source_pure": service_model(pure_training, SERVICE_FEATURES),
            "source_plus_requests": service_model(
                pure_training, SERVICE_FEATURES + REQUESTS
            ),
        }
        if include_regime:
            models["source_plus_regime_requests"] = service_model(
                [{**r, "features": r["regime_features"]} for r in pure_training],
                SERVICE_FEATURES + REQUESTS,
            )
        if instructions is not None:
            models["source_plus_instruction_work"] = service_model(
                [{**r, "features": r["instruction_features"]} for r in pure_training],
                SERVICE_FEATURES + INSTRUCTION_WORK,
            )
        predictions = {name: [] for name in names}
        for row in projected:
            if row["group"] not in excluded:
                continue
            for name in names:
                fallback = name != "source" and not row["pure"]
                predicted = service_prediction(
                    models["source" if fallback else name],
                    row["instruction_features"]
                    if name == "source_plus_instruction_work"
                    else row["regime_features"]
                    if name == "source_plus_regime_requests"
                    else row["features"],
                )
                if name == "source_plus_instruction_work":
                    predicted["ood_reasons"] += row["instruction_ood"]
                elif name == "source_plus_regime_requests":
                    predicted["ood_reasons"] += row["regime_ood"]
                elif name == "source_plus_requests" or fallback:
                    predicted["ood_reasons"] += row["counter_ood"]
                predictions[name].append(
                    dict(
                        id=row["id"],
                        group=row["group"],
                        actual_us=row["latency_us"],
                        source_fallback=fallback,
                        **predicted,
                    )
                )
        cache[key] = dict(
            predictions=predictions,
            diagnostic_models=models,
            counter_training_ids=[r["case"]["id"] for r in counters],
            latency_training_ids=[r["id"] for r in training],
            request_service_training_ids=[r["id"] for r in pure_training],
            **(
                dict(instruction_training_ids=[r["case"]["id"] for r in instructions])
                if instructions is not None
                else {}
            ),
        )
        return cache[key]

    def mape(rows):
        return (
            100
            * sum(abs(r["prediction_us"] / r["actual_us"] - 1) for r in rows)
            / len(rows)
        )

    ordinary = {name: [] for name in names}
    nested, folds = [], []
    for outer in groups:
        scored = fold({outer})
        scores = {
            name: mape(
                [
                    r
                    for inner in groups
                    if inner != outer
                    for r in fold({outer, inner})["predictions"][name]
                    if r["group"] == inner
                ]
            )
            for name in names
        }
        selected = min(scores, key=lambda name: (scores[name], name))
        for name in names:
            ordinary[name] += scored["predictions"][name]
        nested += scored["predictions"][selected]
        folds.append(
            dict(
                held_group=outer,
                selected=selected,
                inner_mape_pct=scores,
                **{k: v for k, v in scored.items() if k != "predictions"},
            )
        )
    return dict(
        role="control",
        eligible_for_fit=False,
        released_model=None,
        count=len(latency_rows),
        ordinary={
            name: dict(mape_pct=mape(rows), rows=rows)
            for name, rows in ordinary.items()
        },
        nested_mape_pct=mape(nested),
        nested_rows=nested,
        folds=folds,
        solver_audit=[
            dict(
                excluded_groups=list(excluded),
                model_solvers={
                    name: model["solver"]
                    for name, model in result["diagnostic_models"].items()
                },
            )
            for excluded, result in sorted(cache.items())
        ],
        caveat="Request service diagnostic, not cache misses or an admitted latency model. Compositions use explicit source fallback; all controls retained. No target artifacts.",
    )
