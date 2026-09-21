import copy

import pytest

from triton_viz.tools.gpu_cold_resource_diagnostic import FEATURES, validate


def fixture():
    resources = [
        dict(
            role="control",
            case=dict(id=f"r{g}_{i}", cv_group=str(g)),
            source_features={k: float(g + i + 1) for k in FEATURES},
            source_precision=[["fp16", "fp16", "ieee"]],
            ood_reasons=[],
            registers_per_thread=32 + g,
            local_bytes_per_thread=4 * g,
        )
        for g in range(4)
        for i in range(2)
    ]
    latencies = [
        {**r, "eligible_for_fit": False, "latency_us": float(10 + i)}
        for i, r in enumerate(resources)
    ]
    return resources, latencies


def test_both_levels_exclude_validation_resources_and_latency_labels():
    resources, latencies = fixture()
    first = validate(resources, latencies)
    mutated_resources, mutated_latencies = copy.deepcopy((resources, latencies))
    for row in mutated_resources:
        if row["case"]["cv_group"] == "0":
            row["registers_per_thread"] = 100000
            row["local_bytes_per_thread"] = 10000000
    for row in mutated_latencies:
        if row["case"]["cv_group"] == "0":
            row["latency_us"] = 99999
    second = validate(mutated_resources, mutated_latencies)
    assert first["folds"][0] == second["folds"][0]
    for name in first["ordinary"]:
        a = [
            (r["id"], r["prediction_us"], r["neighbors"])
            for r in first["ordinary"][name]["rows"]
            if r["group"] == "0"
        ]
        b = [
            (r["id"], r["prediction_us"], r["neighbors"])
            for r in second["ordinary"][name]["rows"]
            if r["group"] == "0"
        ]
        assert a == b
    assert first["count"] == 8 and len(first["nested_rows"]) == 8
    assert first["released_model"] is None and first["eligible_for_fit"] is False
    for fold in first["folds"]:
        assert all(
            not x.startswith("r" + fold["held_group"] + "_")
            for key in ("resource_training_ids", "latency_training_ids")
            for x in fold[key]
        )


def test_diagnostic_refuses_holdout_and_ineligible_protocol_confusion():
    resources, latencies = fixture()
    latencies[0]["eligible_for_fit"] = True
    with pytest.raises(ValueError, match="diagnostic"):
        validate(resources, latencies)
    latencies[0]["eligible_for_fit"] = False
    resources[0]["role"] = "holdout"
    with pytest.raises(ValueError, match="control-only"):
        validate(resources, latencies)


def test_partition_or_source_drift_cannot_leak_compiler_labels():
    resources, latencies = fixture()
    latencies = copy.deepcopy(latencies)
    latencies[0]["case"]["cv_group"] = "another_group"
    with pytest.raises(ValueError, match="partitions"):
        validate(resources, latencies)
