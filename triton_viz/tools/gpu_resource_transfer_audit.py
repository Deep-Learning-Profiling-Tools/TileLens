"""Control-only, geometry-held-out baseline for source-to-resource transfer.

No latency is read. A nearest-control baseline diagnoses whether the current
source descriptors and control coverage transfer to unseen geometries. This is
not a released predictor, a tuned classifier, or a substitute for latency CV.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from triton_viz.tools.gpu_cost_model_pipeline import _write

FEATURES = (
    "logical_dot_accumulator_words_per_thread",
    "logical_dot_operand_words_per_thread",
    "max_float_tile_words_per_thread",
    "requested_stages",
    "threads_per_program",
)
LABELS = ("registers_per_thread", "local_bytes_per_thread")
FEATURE_SETS = {
    "base": FEATURES,
    "live_structure": FEATURES
    + (
        "logical_live_float_words_per_thread",
        "dots_per_program",
        "reductions_per_program",
        "max_dot_m",
        "max_dot_n",
        "max_dot_k",
    ),
}


def structural_features(row):
    """Use only descriptors exported by CPU observation, never case labels."""
    programs = row["program_count"]
    if isinstance(programs, bool) or not isinstance(programs, int) or programs <= 0:
        raise ValueError("Invalid observed program count")
    shapes = row["dot_shapes"]
    if not shapes or any(
        len(pair) != 2 or any(len(s) != 2 for s in pair) or pair[0][1] != pair[1][0]
        for pair in shapes
    ):
        raise ValueError("Unsupported source dot shapes")
    return {
        **row["source_features"],
        **row["source_liveness"],
        "dots_per_program": row["operation_counts"].get("dot", 0) / programs,
        "reductions_per_program": sum(
            count
            for op, count in row["operation_counts"].items()
            if op == "reduce" or op.startswith("reduce_")
        )
        / programs,
        "max_dot_m": max(pair[0][0] for pair in shapes),
        "max_dot_n": max(pair[1][1] for pair in shapes),
        "max_dot_k": max(pair[0][1] for pair in shapes),
    }


def join_control_sources(source_root, resource_root):
    """Join independently observed CPU sources and digest-checked control labels."""
    from triton_viz.tools.gpu_control_resource_audit import audit as resource_audit

    manifest = json.loads((source_root / "manifest.json").read_text())
    if manifest.get("role") != "control":
        raise ValueError("Source manifest must be control-only")
    compiled = resource_audit(resource_root)
    if not compiled["complete"]:
        raise ValueError("Incomplete compiler resource collection")
    source_cases = {case["id"]: case for case in manifest["cases"]}
    if len(source_cases) != len(manifest["cases"]) or set(source_cases) != {
        row["case"]["id"] for row in compiled["rows"]
    }:
        raise ValueError("Source and resource control declarations differ")
    rows = []
    for resource in compiled["rows"]:
        case = resource["case"]
        source = json.loads(
            (source_root / "controls" / (case["id"] + ".json")).read_text()
        )
        if (
            source.get("role") != "control"
            or source["case"] != case
            or source_cases[case["id"]] != case
            or source.get("numerical_validation") != "passed"
            or source.get("compile_and_cuda_forbidden") is not True
        ):
            raise ValueError("Unverified or mismatched source control")
        rows.append({**source, **{key: resource[key] for key in LABELS}})
    return dict(role="control", complete=True, rows=rows)


def descriptor_collisions(rows):
    """Expose indistinguishable descriptors with differing compiler labels."""
    groups = {}
    for row in rows:
        key = json.dumps(
            [row["source_features"], row["source_precision"]], sort_keys=True
        )
        groups.setdefault(key, []).append(row)
    return [
        dict(
            ids=[row["case"]["id"] for row in group],
            label_ranges={
                key: [min(row[key] for row in group), max(row[key] for row in group)]
                for key in LABELS
            },
        )
        for group in groups.values()
        if any(len({row[key] for row in group}) > 1 for key in LABELS)
    ]


def descriptor_error_floor(rows):
    """Empirical MAE lower bound for any deterministic use of these descriptors.

    This is a diagnostic of missing information, not a trained validation score.
    Median is the minimum absolute-error constant for identical inputs.
    """
    groups = {}
    for row in rows:
        key = json.dumps(
            [row["source_features"], row["source_precision"]], sort_keys=True
        )
        groups.setdefault(key, []).append(row)
    return {
        key: sum(
            sum(
                abs(row[key] - statistics.median(r[key] for r in group))
                for row in group
            )
            for group in groups.values()
        )
        / len(rows)
        for key in LABELS
    }


def train(rows, feature_names=FEATURES):
    """Freeze source descriptors and labels from this training fold only."""
    if not rows or any(row.get("role") != "control" for row in rows):
        raise ValueError("Training accepts nonempty control rows only")
    for row in rows:
        values = [row["source_features"][key] for key in feature_names]
        values += [row[key] for key in LABELS]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Invalid resource descriptor or label")
        if row.get("ood_reasons") or not row["source_precision"]:
            raise ValueError("Unsupported source resource descriptor")
    return {
        "feature_names": list(feature_names),
        "domain": {
            key: [
                min(r["source_features"][key] for r in rows),
                max(r["source_features"][key] for r in rows),
            ]
            for key in feature_names
        },
        "controls": [
            {key: row[key] for key in ("source_features", "source_precision", *LABELS)}
            | {"id": row["case"]["id"]}
            for row in rows
        ],
    }


def predict(model, features, precision):
    """Inference gets only source descriptors and a frozen training table."""
    feature_names = model["feature_names"]
    if any(
        not math.isfinite(features[key]) or features[key] < 0 for key in feature_names
    ):
        raise ValueError("Invalid source descriptor")
    reasons = [
        f"outside_training_domain:{key}"
        for key in feature_names
        if not model["domain"][key][0] <= features[key] <= model["domain"][key][1]
    ]
    candidates = [r for r in model["controls"] if r["source_precision"] == precision]
    if not candidates:
        return dict(
            prediction=None, ood_reasons=[*reasons, "unseen_precision"], neighbors=[]
        )
    distances = []
    for row in candidates:
        distance = 0.0
        for key in feature_names:
            lo, hi = model["domain"][key]
            if hi > lo:
                distance += abs(features[key] - row["source_features"][key]) / (hi - lo)
            elif features[key] != lo:
                reasons.append(f"unseen_constant_feature:{key}")
        distances.append(distance)
    minimum = min(distances)
    nearest = [
        row for row, distance in zip(candidates, distances) if distance == minimum
    ]
    return dict(
        prediction={
            key: sum(row[key] for row in nearest) / len(nearest) for key in LABELS
        },
        neighbors=[row["id"] for row in nearest],
        ood_reasons=sorted(set(reasons)),
    )


def audit(report, *, feature_set="base"):
    if report.get("role") != "control" or not report.get("complete"):
        raise ValueError("A complete control-only resource audit is required")
    if feature_set not in FEATURE_SETS:
        raise ValueError("Unknown source descriptor set")
    # Explicit projection: never copy/read HES latency values into training.
    rows = [
        dict(
            role="control",
            **{
                key: r[key]
                for key in (
                    "case",
                    "source_features",
                    "source_precision",
                    "ood_reasons",
                    *LABELS,
                )
            },
        )
        for r in report["rows"]
    ]
    if feature_set == "live_structure":
        for row, original in zip(rows, report["rows"]):
            row["source_features"] = structural_features(original)
    groups = sorted({r["case"]["cv_group"] for r in rows})
    if len(groups) < 3 or len({r["case"]["id"] for r in rows}) != len(rows):
        raise ValueError("Need unique controls in at least three geometry groups")
    results = []
    for group in groups:
        training = [r for r in rows if r["case"]["cv_group"] != group]
        model = train(training, FEATURE_SETS[feature_set])
        for row in rows:
            if row["case"]["cv_group"] != group:
                continue
            result = predict(model, row["source_features"], row["source_precision"])
            results.append(
                dict(
                    id=row["case"]["id"],
                    group=group,
                    training_ids=[r["case"]["id"] for r in training],
                    actual={key: row[key] for key in LABELS},
                    **result,
                )
            )
    supported = [r for r in results if r["prediction"] is not None]
    return dict(
        role="control",
        feature_set=feature_set,
        eligible_for_fit=False,
        count=len(results),
        descriptor_collisions=descriptor_collisions(rows),
        descriptor_empirical_mae_floor=descriptor_error_floor(rows),
        rows=results,
        protocol="leave-one-declared-geometry-group-out; train-range normalized L1 nearest control; exact precision; average distance ties",
        resource_mae={
            key: sum(abs(r["prediction"][key] - r["actual"][key]) for r in supported)
            / len(supported)
            if supported
            else None
            for key in LABELS
        },
        unknown_count=len(results) - len(supported),
        ood_count=sum(bool(r["ood_reasons"]) for r in results),
        allocation_confusion={
            name: sum(
                (r["actual"]["local_bytes_per_thread"] > 0) == actual
                and (r["prediction"]["local_bytes_per_thread"] > 0) == predicted
                for r in supported
            )
            for name, actual, predicted in [
                ("true_positive", True, True),
                ("false_negative", True, False),
                ("false_positive", False, True),
                ("true_negative", False, False),
            ]
        },
        caveat="Local allocation is not dynamic spill traffic. No latency fit, candidate search, gate pass, or target evaluation is implied.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--controls", type=Path)
    inputs.add_argument("--source-root", type=Path)
    parser.add_argument("--resource-root", type=Path)
    parser.add_argument("--feature-set", choices=tuple(FEATURE_SETS), default="base")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Use a new output path")
    if args.source_root:
        if args.resource_root is None:
            parser.error("--source-root requires --resource-root")
        controls = join_control_sources(args.source_root, args.resource_root)
    else:
        if args.resource_root is not None:
            parser.error("--resource-root requires --source-root")
        controls = json.loads(args.controls.read_text())
    result = audit(controls, feature_set=args.feature_set)
    _write(args.output, result)
    print(
        {
            key: result[key]
            for key in ("count", "resource_mae", "allocation_confusion", "ood_count")
        }
    )


if __name__ == "__main__":
    main()
