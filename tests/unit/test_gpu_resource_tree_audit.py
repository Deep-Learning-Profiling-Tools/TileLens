import copy

import pytest

from triton_viz.tools.gpu_resource_tree_audit import FEATURES, predict, train, validate


def controls():
    return [
        dict(
            role="control",
            case=dict(id=f"{g}_{s}_{i}", cv_group=str(g)),
            source_features={**dict.fromkeys(FEATURES, 1), "requested_stages": s},
            source_precision=[["fp32", "fp32", "ieee"]],
            ood_reasons=[],
            local_bytes_per_thread=100 if s == 1 else 0,
            registers_per_thread=40 if s == 1 else 255,
        )
        for g in range(3)
        for s in (1, 2)
        for i in range(2)
    ]


def test_source_resource_mapping_can_represent_nonmonotonic_stage_effect():
    rows = controls()
    result = validate(rows)
    assert result["count"] == 12
    assert result["resource_mae"]["local_bytes_per_thread"] == pytest.approx(
        0, abs=1e-12
    )
    assert result["allocation_confusion"] == dict(tp=6, tn=6, fp=0, fn=0)
    assert result["released_model"] is None
    for row in result["rows"]:
        fold = next(f for f in result["folds"] if f["held_group"] == row["group"])
        assert row["id"] not in fold["training_ids"]


@pytest.mark.parametrize("feature_set", ["initial_layout", "mma_materialization"])
def test_outer_labels_do_not_change_fold_model_or_hyperparameter_selection(feature_set):
    from triton_viz.tools.gpu_resource_transfer_audit import FEATURE_SETS

    rows = controls()
    names = FEATURE_SETS[feature_set]
    for row in rows:
        row["source_features"] = {**dict.fromkeys(names, 1), **row["source_features"]}
    first = validate(rows, feature_names=names)["folds"][0]
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["case"]["cv_group"] == "0":
            row["local_bytes_per_thread"] = 9999
            row["registers_per_thread"] = 1
    second = validate(changed, feature_names=names)["folds"][0]
    assert first == second


def test_inference_uses_only_source_and_flags_domain_without_reading_labels():
    rows = controls()
    model = train(rows, depth=2, leaf=1)
    features = dict(rows[0]["source_features"], requested_stages=4)
    result = predict(model, features, rows[0]["source_precision"])
    assert "outside_training_domain:requested_stages" in result["ood_reasons"]
    assert predict(model, features, [["fp16", "fp16", "ieee"]])["prediction"] is None
    rows[0]["role"] = "holdout"
    with pytest.raises(ValueError):
        train(rows, depth=2, leaf=1)


def test_leaf_prediction_uses_the_same_log_space_as_split_objective():
    rows = controls()
    model = train(rows, depth=0, leaf=1)
    result = predict(model, rows[0]["source_features"], rows[0]["source_precision"])
    assert result["prediction"]["local_bytes_per_thread"] == pytest.approx(101**0.5 - 1)


def test_shared_geometries_and_declared_groups_cannot_leak_across_suites():
    from triton_viz.tools.gpu_resource_tree_audit import source_geometry_partition

    rows = [
        dict(case=dict(id=str(i), cv_group=g), dot_shapes=shapes)
        for i, (g, shapes) in enumerate(
            [
                ("suite_a", [[[32, 16], [16, 64]]]),
                ("suite_b", [[[32, 16], [16, 64]], [[32, 64], [64, 32]]]),
                ("suite_c", [[[32, 64], [64, 32]]]),
                ("suite_c", [[[16, 16], [16, 16]]]),
                ("independent", [[[128, 64], [64, 256]]]),
            ]
        )
    ]
    groups = source_geometry_partition(rows)
    assert len({groups[str(i)] for i in range(4)}) == 1
    assert groups["4"] != groups["0"]
