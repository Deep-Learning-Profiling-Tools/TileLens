import pytest

from microbench.gpu.common.cache_controls import cache_declaration
from triton_viz.tools.gpu_cache_counter_audit import audit_ranges
from triton_viz.tools.gpu_cache_counter_model_audit import audit, copy_range_misses


def test_capacity_declaration_preserves_old_controls_and_adds_independent_groups():
    old, new = cache_declaration(), cache_declaration("capacity")
    assert set(old["working_set_mib"]) <= set(new["working_set_mib"])
    assert len(new["working_set_mib"]) == 9
    assert new["l2_bytes"] == 25165824
    with pytest.raises(ValueError):
        cache_declaration("holdout")


def test_missing_capacity_counter_rows_are_never_deleted(tmp_path):
    report = audit_ranges(tmp_path, "capacity")
    assert not report["complete"] and not report["eligible_for_latency_fit"]
    assert len(report["rows"]) == 27
    assert len({r["cv_group"] for r in report["rows"]}) == 9


def test_capacity_sensitivity_retains_all_rows_and_never_selects_coefficients():
    declaration = cache_declaration("capacity")
    rows = [
        dict(
            working_set_mib=mib,
            eviction=eviction,
            counters=dict(
                misses=copy_range_misses(
                    mib * 1024**2,
                    eviction,
                    l2_bytes=declaration["l2_bytes"],
                    tile_bytes=32,
                    associativity=4,
                )
            ),
        )
        for mib in declaration["working_set_mib"]
        for eviction in declaration["evictions"]
    ]
    report = dict(
        role="control",
        matrix="capacity",
        complete=True,
        replay_counts_consistent=True,
        rows=rows,
    )
    result = audit(
        report, l2_bytes=declaration["l2_bytes"], associativities=[1, 4], tile_bytes=32
    )
    assert result["selected"] is None and not result["eligible_for_fit"]
    assert all(len(s["rows"]) == 27 for s in result["scenarios"])
    assert result["scenarios"][1]["miss_count_mape_pct"] == pytest.approx(0)
    report["rows"].pop()
    with pytest.raises(ValueError, match="capacity-control"):
        audit(report, l2_bytes=declaration["l2_bytes"], associativities=[4])
