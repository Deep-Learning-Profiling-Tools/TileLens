"""Nested control-only latency diagnostic; never a calibrated/released model.

Compare source descriptors with/without predicted allocation using either
train-range-normalized nearest controls or nonnegative service pricing.
Every resource fit excludes the latency validation geometry, including inner
folds. No admission flags change and no production calibration is emitted.
"""

import math

import numpy as np

from triton_viz.performance.calibration import _nonnegative_fit
from triton_viz.performance.gpu_dot_precision import WAVE_DOT_FEATURES

from triton_viz.tools import gpu_resource_transfer_audit as resources


FEATURES = resources.FEATURE_SETS["initial_layout"]
EXTRA = ("predicted_registers_per_thread", "predicted_local_bytes_per_thread")
SERVICE_FEATURES = (
    "launch",
    "global_sectors",
    "alu_warps",
    "sfu_warps",
    "shuffle_steps",
) + WAVE_DOT_FEATURES
DEMAND = "predicted_local_allocation_dot_demand"


def allocation_dot_demand(source_features, waves, allocation_bytes):
    """Allocation-volume/episode proxy, not a count of dynamic local accesses."""
    threads = source_features["threads_per_program"]
    dots = source_features["dots_per_program"]
    if (
        any(
            isinstance(v, bool) or not math.isfinite(v) or v < 0
            for v in (threads, dots, waves, allocation_bytes)
        )
        or threads == 0
        or waves == 0
    ):
        raise ValueError("Invalid source allocation demand")
    return allocation_bytes * threads * dots * waves


def service_model(training, keys):
    """Diagnostic math only; no admission decision or production calibration."""
    x = np.array([[r["features"][k] for k in keys] for r in training], dtype=float)
    y = np.array([r["latency_us"] for r in training], dtype=float)
    if (
        not len(training)
        or not np.isfinite(x).all()
        or (x < 0).any()
        or not np.isfinite(y).all()
        or (y <= 0).any()
    ):
        raise ValueError("Invalid diagnostic service data")
    return dict(
        keys=list(keys),
        coefficients=_nonnegative_fit(x, y).tolist(),
        domain={
            k: [float(x[:, i].min()), float(x[:, i].max())] for i, k in enumerate(keys)
        },
    )


def service_prediction(model, features):
    values = [features[k] for k in model["keys"]]
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Invalid service prediction features")
    return dict(
        prediction_us=sum(v * c for v, c in zip(values, model["coefficients"])),
        ood_reasons=[
            f"outside_training_domain:{k}"
            for k in model["keys"]
            if not model["domain"][k][0] <= features[k] <= model["domain"][k][1]
        ],
    )


def nearest(training, features, precision, keys):
    if not training:
        raise ValueError("Empty diagnostic training fold")
    domain = {
        k: (
            min(r["features"][k] for r in training),
            max(r["features"][k] for r in training),
        )
        for k in keys
    }
    reasons = [
        f"outside_training_domain:{k}"
        for k in keys
        if not domain[k][0] <= features[k] <= domain[k][1]
    ]
    candidates = [r for r in training if r["source_precision"] == precision]
    if not candidates:
        raise ValueError("Unseen precision; retain controls and fail diagnostic")
    distances = [
        sum(
            abs(features[k] - r["features"][k]) / (hi - lo)
            if hi > lo
            else float(features[k] != lo)
            for k, (lo, hi) in domain.items()
        )
        for r in candidates
    ]
    best = min(distances)
    neighbors = [r for r, d in zip(candidates, distances) if d == best]
    return dict(
        prediction_us=sum(r["latency_us"] for r in neighbors) / len(neighbors),
        neighbors=[r["id"] for r in neighbors],
        ood_reasons=reasons,
    )


def validate(resource_rows, latency_rows, *, pricing="nearest"):
    if pricing not in {"nearest", "service"}:
        raise ValueError("Unknown diagnostic pricing rule")
    for rows in (resource_rows, latency_rows):
        if not rows or any(r.get("role") != "control" for r in rows):
            raise ValueError("Diagnostic requires nonempty control-only inputs")
        if len({r["case"]["id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate controls")
    resource_by_id = {r["case"]["id"]: r for r in resource_rows}
    for row in latency_rows:
        matching = resource_by_id.get(row["case"]["id"])
        if (
            matching is None
            or matching["case"] != row["case"]
            or matching["source_features"] != row["source_features"]
            or matching["source_precision"] != row["source_precision"]
        ):
            raise ValueError("Require matching source controls and geometry partitions")
        if row.get("eligible_for_fit") is not False:
            raise ValueError("Require explicitly diagnostic timing rows")
        if not math.isfinite(row["latency_us"]) or row["latency_us"] <= 0:
            raise ValueError("Invalid control timing")
        if any(
            not math.isfinite(row["source_features"][k])
            or row["source_features"][k] < 0
            for k in FEATURES
        ):
            raise ValueError("Invalid source descriptor")
        if pricing == "service" and any(
            not math.isfinite(row["pricing_features"][k])
            or row["pricing_features"][k] < 0
            for k in (*SERVICE_FEATURES, "waves")
        ):
            raise ValueError("Invalid source service features")
    groups = sorted({r["case"]["cv_group"] for r in latency_rows})
    if len(groups) < 3:
        raise ValueError("Nested diagnostic needs three geometry groups")
    cache = {}

    def fold(excluded):
        key = tuple(sorted(excluded))
        if key in cache:
            return cache[key]
        resource_training = [
            r for r in resource_rows if r["case"]["cv_group"] not in excluded
        ]
        model = resources.train(resource_training, FEATURES)
        projected = []
        for row in latency_rows:
            result = resources.predict(
                model, row["source_features"], row["source_precision"]
            )
            if result["prediction"] is None:
                raise ValueError("Unscored resource prediction; no control deletion")
            projected.append(
                dict(
                    id=row["case"]["id"],
                    group=row["case"]["cv_group"],
                    source_precision=row["source_precision"],
                    latency_us=row["latency_us"],
                    resource_ood=result["ood_reasons"],
                    features={
                        **row["source_features"],
                        **{
                            f"predicted_{k}": v for k, v in result["prediction"].items()
                        },
                    },
                )
            )
            if pricing == "service":
                work = row["pricing_features"]
                projected[-1]["features"] = {
                    **{k: work[k] for k in SERVICE_FEATURES},
                    DEMAND: allocation_dot_demand(
                        row["source_features"],
                        work["waves"],
                        result["prediction"]["local_bytes_per_thread"],
                    ),
                }
        training = [r for r in projected if r["group"] not in excluded]
        candidates = (
            (("source", FEATURES), ("source_plus_resource", FEATURES + EXTRA))
            if pricing == "nearest"
            else (
                ("source", SERVICE_FEATURES),
                ("source_plus_resource", SERVICE_FEATURES + (DEMAND,)),
            )
        )
        models = (
            {name: service_model(training, keys) for name, keys in candidates}
            if pricing == "service"
            else {}
        )
        predictions = {name: [] for name in ("source", "source_plus_resource")}
        for row in projected:
            if row["group"] not in excluded:
                continue
            for name, keys in candidates:
                result = (
                    service_prediction(models[name], row["features"])
                    if pricing == "service"
                    else nearest(
                        training, row["features"], row["source_precision"], keys
                    )
                )
                if name == "source_plus_resource":
                    result["ood_reasons"] += [
                        "resource:" + r for r in row["resource_ood"]
                    ]
                predictions[name].append(
                    dict(
                        id=row["id"],
                        group=row["group"],
                        actual_us=row["latency_us"],
                        **result,
                    )
                )
        cache[key] = dict(
            diagnostic_service_models=models,
            predictions=predictions,
            resource_training_ids=[r["case"]["id"] for r in resource_training],
            latency_training_ids=[r["id"] for r in training],
        )
        return cache[key]

    def mape(rows):
        return (
            100
            * sum(abs(r["prediction_us"] / r["actual_us"] - 1) for r in rows)
            / len(rows)
        )

    ordinary = {name: [] for name in ("source", "source_plus_resource")}
    nested, folds = [], []
    for outer in groups:
        scored = fold({outer})
        scores = {}
        for name in ordinary:
            ordinary[name].extend(scored["predictions"][name])
            inner_rows = [
                r
                for inner in groups
                if inner != outer
                for r in fold({outer, inner})["predictions"][name]
                if r["group"] == inner
            ]
            scores[name] = mape(inner_rows)
        selected = min(scores, key=lambda name: (scores[name], name))
        nested.extend(scored["predictions"][selected])
        folds.append(
            dict(
                held_group=outer,
                inner_mape_pct=scores,
                selected=selected,
                diagnostic_service_models=scored["diagnostic_service_models"],
                resource_training_ids=scored["resource_training_ids"],
                latency_training_ids=scored["latency_training_ids"],
            )
        )
    return dict(
        role="control",
        count=len(latency_rows),
        eligible_for_fit=False,
        released_model=None,
        pricing=pricing,
        ordinary={
            name: dict(mape_pct=mape(rows), rows=rows)
            for name, rows in ordinary.items()
        },
        nested_mape_pct=mape(nested),
        nested_rows=nested,
        folds=folds,
        caveat="Unadmitted timing diagnostic, not a release gate. Predicted allocation is not dynamic spill traffic or cache misses; no holdout read or protocol relabeling.",
    )
