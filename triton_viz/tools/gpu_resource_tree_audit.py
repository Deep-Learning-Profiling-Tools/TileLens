"""Fold-local nonlinear source-to-resource diagnostic, never latency admission."""

import argparse
import json
import math
from pathlib import Path

from triton_viz.tools.gpu_cost_model_pipeline import _write
from triton_viz.tools import gpu_resource_transfer_audit as baseline

FEATURES = baseline.FEATURE_SETS["initial_layout"]
SCENARIOS = tuple((depth, leaf) for depth in (2, 4, 8) for leaf in (1, 2, 4))


def train(rows, *, depth, leaf):
    if (
        isinstance(depth, bool)
        or not isinstance(depth, int)
        or depth < 0
        or isinstance(leaf, bool)
        or not isinstance(leaf, int)
        or leaf < 1
    ):
        raise ValueError("Invalid tree limits")
    checked = baseline.train(rows, FEATURES)

    def build(group, remaining):
        values = [math.log1p(r["local_bytes_per_thread"]) for r in group]
        mean = sum(values) / len(values)
        cost = sum((v - mean) ** 2 for v in values)
        node = dict(
            prediction={
                k: sum(r[k] for r in group) / len(group) for k in baseline.LABELS
            }
        )
        # Split loss is squared log1p-local error; its leaf estimator must
        # inhabit that same space, not silently switch to a raw-byte mean.
        node["prediction"]["local_bytes_per_thread"] = math.expm1(mean)
        if remaining == 0 or len(group) < 2 * leaf or cost == 0:
            return node
        best = None
        for key in FEATURES:
            unique = sorted({r["source_features"][key] for r in group})
            for a, b in zip(unique, unique[1:]):
                threshold = (a + b) / 2
                left = [r for r in group if r["source_features"][key] <= threshold]
                right = [r for r in group if r["source_features"][key] > threshold]
                if min(len(left), len(right)) < leaf:
                    continue
                loss = 0
                for side in (left, right):
                    targets = [math.log1p(r["local_bytes_per_thread"]) for r in side]
                    center = sum(targets) / len(targets)
                    loss += sum((v - center) ** 2 for v in targets)
                candidate = (loss, key, threshold)
                if loss < cost and (best is None or candidate < best[0]):
                    best = (candidate, left, right)
        if best is not None:
            (_, key, threshold), left, right = best
            node.update(
                feature=key,
                threshold=threshold,
                left=build(left, remaining - 1),
                right=build(right, remaining - 1),
            )
        return node

    strata = {}
    for row in rows:
        strata.setdefault(json.dumps(row["source_precision"]), []).append(row)
    return dict(
        feature_names=list(FEATURES),
        domain=checked["domain"],
        trees={key: build(group, depth) for key, group in strata.items()},
    )


def predict(model, features, precision):
    # This API cannot read a case ID, compiler artifact, label or latency.
    if model.get("feature_names") != list(FEATURES):
        raise ValueError("Resource feature schema mismatch")
    if any(not math.isfinite(features[k]) or features[k] < 0 for k in FEATURES):
        raise ValueError("Invalid source features")
    reasons = [
        f"outside_training_domain:{k}"
        for k in FEATURES
        if not model["domain"][k][0] <= features[k] <= model["domain"][k][1]
    ]
    node = model["trees"].get(json.dumps(precision))
    if node is None:
        return dict(prediction=None, ood_reasons=[*reasons, "unseen_precision"])
    while "feature" in node:
        node = (
            node["left"]
            if features[node["feature"]] <= node["threshold"]
            else node["right"]
        )
    return dict(prediction=node["prediction"], ood_reasons=reasons)


def validate(rows):
    groups = sorted({r["case"]["cv_group"] for r in rows})
    if len(groups) < 3 or len({r["case"]["id"] for r in rows}) != len(rows):
        raise ValueError("Require unique controls in three or more groups")
    results, folds = [], []
    for outer in groups:
        training = [r for r in rows if r["case"]["cv_group"] != outer]
        scores = []
        for depth, leaf in SCENARIOS:
            errors = []
            for inner in groups:
                if inner == outer:
                    continue
                fit_rows = [r for r in training if r["case"]["cv_group"] != inner]
                model = train(fit_rows, depth=depth, leaf=leaf)
                for row in training:
                    if row["case"]["cv_group"] != inner:
                        continue
                    result = predict(
                        model, row["source_features"], row["source_precision"]
                    )
                    if result["prediction"] is None:
                        raise ValueError(
                            "Unseen precision retained; validation cannot score all controls"
                        )
                    errors.append(
                        abs(
                            math.log1p(result["prediction"]["local_bytes_per_thread"])
                            - math.log1p(row["local_bytes_per_thread"])
                        )
                    )
            scores.append(
                dict(depth=depth, leaf=leaf, log1p_local_mae=sum(errors) / len(errors))
            )
        selected = min(
            scores, key=lambda r: (r["log1p_local_mae"], r["depth"], -r["leaf"])
        )
        model = train(training, depth=selected["depth"], leaf=selected["leaf"])
        folds.append(
            dict(
                held_group=outer,
                inner_scores=scores,
                selected=selected,
                training_ids=[r["case"]["id"] for r in training],
                model=model,
            )
        )
        for row in rows:
            if row["case"]["cv_group"] == outer:
                result = predict(model, row["source_features"], row["source_precision"])
                if result["prediction"] is None:
                    raise ValueError("Unscored control; no deletion allowed")
                results.append(
                    dict(
                        id=row["case"]["id"],
                        group=outer,
                        actual={k: row[k] for k in baseline.LABELS},
                        **result,
                    )
                )
    confusion = dict(tp=0, fn=0, fp=0, tn=0)
    for row in results:
        actual = row["actual"]["local_bytes_per_thread"] > 0
        predicted = row["prediction"]["local_bytes_per_thread"] > 0
        confusion[
            ("tp" if predicted else "fn") if actual else ("fp" if predicted else "tn")
        ] += 1
    return dict(
        role="control",
        eligible_for_fit=False,
        released_model=None,
        count=len(results),
        folds=folds,
        rows=results,
        resource_mae={
            k: sum(abs(r["prediction"][k] - r["actual"][k]) for r in results)
            / len(results)
            for k in baseline.LABELS
        },
        allocation_confusion=confusion,
        ood_count=sum(bool(r["ood_reasons"]) for r in results),
        caveat="Resource-label nested validation only. Local allocation is not dynamic spill traffic; no latency labels, latency gates or target evaluation.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, nargs="+", required=True)
    parser.add_argument("--resource-root", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Require fresh output")
    if len(args.source_root) != len(args.resource_root):
        raise ValueError("Pair every source collection with its compiler collection")
    reports = [
        baseline.join_control_sources(source, resource)
        for source, resource in zip(args.source_root, args.resource_root)
    ]
    if len({r["compiler_version"] for r in reports}) != 1:
        raise ValueError("Do not mix compiler policy versions")
    # Do not silently mix explicit backend stack labels with a fallback inferred
    # from Triton's spill field; their equivalence needs independent validation.
    for root, report in zip(args.resource_root, reports):
        for row in report["rows"]:
            raw = json.loads(
                (root / "controls" / (row["case"]["id"] + ".json")).read_text()
            )
            if "local_bytes_per_thread" not in raw:
                raise ValueError(
                    "Require explicit backend local allocation labels for every control"
                )
    report = dict(
        role="control",
        complete=True,
        compiler_version=reports[0]["compiler_version"],
        rows=[row for collection in reports for row in collection["rows"]],
    )
    # Explicit projection excludes any timing and target fields.
    rows = [
        dict(
            role="control",
            case=r["case"],
            source_precision=r["source_precision"],
            ood_reasons=r["ood_reasons"],
            source_features=baseline.layout_features(
                r, compiler_version=report["compiler_version"]
            ),
            **{k: r[k] for k in baseline.LABELS},
        )
        for r in report["rows"]
    ]
    result = validate(rows)
    result["compiler_version"] = report["compiler_version"]
    reference = baseline.audit(report, feature_set="initial_layout")
    result["nearest_control_baseline"] = {
        k: reference[k]
        for k in ("count", "resource_mae", "allocation_confusion", "ood_count")
    }
    _write(args.output, result)
    print(
        {
            k: result[k]
            for k in ("count", "resource_mae", "allocation_confusion", "ood_count")
        }
    )


if __name__ == "__main__":
    main()
