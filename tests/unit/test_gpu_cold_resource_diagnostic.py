import copy

import numpy as np
import pytest

from triton_viz.tools.gpu_cold_resource_diagnostic import (
    FEATURES,
    SERVICE_FEATURES,
    validate,
    enumerated_nonnegative_fit,
    service_model,
)


def test_small_nnls_handles_collinear_and_inactive_columns_without_dropping_rows():
    x = np.array([[1, 2, 0, 1], [2, 4, 0, 1], [3, 6, 0, 1]], dtype=float)
    y = np.array([4, 7, 10], dtype=float)
    coefficients = enumerated_nonnegative_fit(x, y)
    assert (coefficients >= 0).all()
    assert np.allclose(x @ coefficients, y)
    assert coefficients[2] == 0
    # The unconstrained slope here is negative: NNLS must use the boundary.
    x = np.array([[1, 1], [1, 2], [1, 3]], dtype=float)
    y = np.array([3, 2, 1], dtype=float)
    coefficients = enumerated_nonnegative_fit(x, y)
    assert coefficients[1] == 0
    assert coefficients[0] == pytest.approx(sum(1 / y) / sum(1 / y**2))


def test_service_solver_fallback_preserves_objective_and_records_provenance(
    monkeypatch,
):
    from triton_viz.tools import gpu_cold_resource_diagnostic as diagnostic

    def fail(*_):
        raise ValueError("Nonnegative calibration did not converge")

    monkeypatch.setattr(diagnostic, "_nonnegative_fit", fail)
    rows = [dict(features={"a": i, "b": 2 * i}, latency_us=3 * i) for i in (1, 2, 3)]
    model = service_model(rows, ("a", "b"))
    assert model["solver"] == "enumerated_faces_svd"
    assert sum(model["coefficients"] * np.array([1, 2])) == pytest.approx(3)


def test_enumerated_solver_satisfies_nonnegative_optimality_conditions():
    rng = np.random.default_rng(20260921)
    for _ in range(12):
        x = rng.uniform(size=(20, 5)) * np.array([1, 1e-9, 1e9, 1, 1])
        x[:, 4] = x[:, 3]  # Non-unique coefficients must not break predictions.
        y = rng.uniform(1, 10, size=20)
        coefficients = enumerated_nonnegative_fit(x, y)
        weighted = x / y[:, None]
        scale = np.linalg.norm(weighted, axis=0)
        a = weighted / scale
        c = coefficients * scale
        gradient = a.T @ (a @ c - 1)
        assert (c >= 0).all()
        assert gradient.min() >= -1e-10
        assert np.max(np.abs(c * gradient)) < 1e-10


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


@pytest.mark.parametrize("pricing", ["nearest", "service"])
def test_both_levels_exclude_validation_resources_and_latency_labels(pricing):
    resources, latencies = fixture()
    for row in latencies:
        row["pricing_features"] = {
            **dict.fromkeys(SERVICE_FEATURES, 0),
            "launch": 1,
            "waves": 1,
        }
    first = validate(resources, latencies, pricing=pricing)
    mutated_resources, mutated_latencies = copy.deepcopy((resources, latencies))
    for row in mutated_resources:
        if row["case"]["cv_group"] == "0":
            row["registers_per_thread"] = 100000
            row["local_bytes_per_thread"] = 10000000
    for row in mutated_latencies:
        if row["case"]["cv_group"] == "0":
            row["latency_us"] = 99999
    second = validate(mutated_resources, mutated_latencies, pricing=pricing)
    assert first["folds"][0] == second["folds"][0]
    for name in first["ordinary"]:
        a = [
            (r["id"], r["prediction_us"], r.get("neighbors"))
            for r in first["ordinary"][name]["rows"]
            if r["group"] == "0"
        ]
        b = [
            (r["id"], r["prediction_us"], r.get("neighbors"))
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


def test_allocation_demand_has_explicit_units_and_no_fitted_constant():
    from triton_viz.tools.gpu_cold_resource_diagnostic import allocation_dot_demand

    source = dict(threads_per_program=128, dots_per_program=5)
    assert allocation_dot_demand(source, 2, 64) == 81920
    assert allocation_dot_demand(source, 2, 0) == 0
    for invalid in (-1, float("nan"), True):
        with pytest.raises(ValueError):
            allocation_dot_demand(source, 2, invalid)
    with pytest.raises(ValueError):
        allocation_dot_demand(source, 0, 64)
