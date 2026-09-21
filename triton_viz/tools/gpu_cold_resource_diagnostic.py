"""Nested control-only latency diagnostic; never a calibrated/released model.

Compare source descriptors with/without predicted allocation using the same
train-range-normalized nearest-control rule. Every resource fit excludes the
latency validation geometry, including inner folds. No admission flags change.
"""

import math

from triton_viz.tools import gpu_resource_transfer_audit as resources


FEATURES = resources.FEATURE_SETS["initial_layout"]
EXTRA = ("predicted_registers_per_thread", "predicted_local_bytes_per_thread")


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


def validate(resource_rows, latency_rows):
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
        training = [r for r in projected if r["group"] not in excluded]
        predictions = {name: [] for name in ("source", "source_plus_resource")}
        for row in projected:
            if row["group"] not in excluded:
                continue
            for name, keys in (
                ("source", FEATURES),
                ("source_plus_resource", FEATURES + EXTRA),
            ):
                result = nearest(
                    training, row["features"], row["source_precision"], keys
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
                resource_training_ids=scored["resource_training_ids"],
                latency_training_ids=scored["latency_training_ids"],
            )
        )
    return dict(
        role="control",
        count=len(latency_rows),
        eligible_for_fit=False,
        released_model=None,
        ordinary={
            name: dict(mape_pct=mape(rows), rows=rows)
            for name, rows in ordinary.items()
        },
        nested_mape_pct=mape(nested),
        nested_rows=nested,
        folds=folds,
        caveat="Unadmitted timing diagnostic, not a release gate. Predicted allocation is not dynamic spill traffic or cache misses; no holdout read or protocol relabeling.",
    )
