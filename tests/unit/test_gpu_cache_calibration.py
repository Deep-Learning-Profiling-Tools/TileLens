import pytest

from microbench.gpu.common.cache_controls import cache_declaration
from triton_viz.tools.gpu_cache_calibration import fit
from triton_viz.tools.gpu_cache_counter_model_audit import copy_range_misses


def report():
    declaration = cache_declaration("capacity")
    return dict(
        role="control",
        matrix="capacity",
        complete=True,
        replay_counts_consistent=True,
        declaration=declaration,
        rows=[
            dict(
                working_set_mib=mib,
                eviction=eviction,
                status="complete",
                cv_group=f"cache_capacity_mib{mib}",
                latency=object(),
                counters=dict(
                    misses=copy_range_misses(
                        mib * 1024**2,
                        eviction,
                        l2_bytes=declaration["l2_bytes"],
                        tile_bytes=32,
                        associativity=8,
                    )
                ),
            )
            for mib in declaration["working_set_mib"]
            for eviction in declaration["evictions"]
        ],
    )


def test_effective_cache_calibration_refits_every_fold_and_never_fits_latency():
    result = fit(report())
    assert result["count"] == 27 and result["group_count"] == 9
    assert result["dual_counter_cv_gate_passed"]
    assert result["nested_cv_mape_pct"] == pytest.approx(0)
    assert result["released_counter_model"]["effective_associativity"] == 8
    assert not result["eligible_for_latency_fit"]
    for row in result["nested_rows"]:
        assert len(row["training_ids"]) == 24
        assert row["id"] not in row["training_ids"]
        assert all(
            not key.startswith(row["id"].split("_")[0] + "_")
            for key in row["training_ids"]
        )


def test_nested_parameter_and_method_selection_cannot_read_outer_counters():
    original = report()
    baseline = fit(original)
    for row in original["rows"]:
        if row["working_set_mib"] == 12:
            row["counters"]["misses"] *= 17
    changed = fit(original)
    for before, after in zip(baseline["nested_rows"], changed["nested_rows"]):
        if before["group"] == "cache_capacity_mib12":
            for field in (
                "method",
                "associativity",
                "predicted_miss_sectors",
                "training_ids",
                "inner_cv_mape_pct",
            ):
                assert before[field] == after[field]


def test_failed_counter_gate_never_releases_a_model():
    data = report()
    for row in data["rows"]:
        row["counters"]["misses"] = 1
    result = fit(data)
    assert not result["dual_counter_cv_gate_passed"]
    assert result["released_counter_model"] is None
    assert len(result["nested_rows"]) == 27


@pytest.mark.parametrize("change", ["holdout", "missing", "declaration", "nonfinite"])
def test_only_complete_declared_control_counters_are_admitted(change):
    data = report()
    if change == "holdout":
        data["role"] = "holdout"
    elif change == "missing":
        data["rows"].pop()
    elif change == "declaration":
        data["declaration"]["associativity_candidates"] = [8]
    else:
        data["rows"][0]["counters"]["misses"] = float("nan")
    with pytest.raises(ValueError):
        fit(data)
