import json

import pytest

from triton_viz.tools.gpu_cta_schedule_audit import interval_overlap


def test_overlap_counts_half_open_boundaries_and_idle_gaps():
    result = interval_overlap([(1, 4), (4, 7), (9, 12)])
    assert result["peak_overlapping_intervals"] == 1
    assert result["overlap_duration_ns"] == {"0": 2, "1": 9}
    assert result["time_weighted_overlap"] == pytest.approx(9 / 11)
    assert result["common_overlap_fraction"] == 0


def test_all_programs_can_overlap_without_serial_sm_waves():
    result = interval_overlap([(1, 11)] * 384)
    assert result["peak_overlapping_intervals"] == 384
    assert result["time_weighted_overlap"] == 384
    assert result["common_overlap_fraction"] == 1


@pytest.mark.parametrize("intervals", [[], [(0, 1)], [(2, 2)], [(2, 1)], [(1.5, 2)]])
def test_invalid_intervals_rejected(intervals):
    with pytest.raises(ValueError):
        interval_overlap(intervals)


def test_endpoint_migration_is_retained_without_false_per_sm_claim(
    tmp_path, monkeypatch
):
    from triton_viz.tools import gpu_cta_schedule_audit as module

    monkeypatch.setattr(module, "audit_grid", lambda root: None)
    monkeypatch.setattr(
        module, "footprint_grid", lambda: [dict(programs=2, local_slots=32)]
    )
    stem = "local_s32_p2_i65536"
    (tmp_path / (stem + ".log")).write_text("body,0,0,1,10\nbody,0,1,2,11\n")
    path = tmp_path / (stem + ".json")
    path.write_text(json.dumps(dict(native_audit={})))
    assert module.audit_schedule(tmp_path)["rows"][0]["sm_observed"] is False
    observations = [dict(start_sm=94, end_sm=94), dict(start_sm=94, end_sm=90)]
    path.write_text(json.dumps(dict(native_audit=dict(sm_observations=observations))))
    row = module.audit_schedule(tmp_path)["rows"][0]
    assert row["changed_endpoint_ctas"] == [1]
    assert row["per_sm"] is None
    assert row["global_overlap"]["interval_count"] == 2
    observations[1]["end_sm"] = 94
    path.write_text(json.dumps(dict(native_audit=dict(sm_observations=observations))))
    row = module.audit_schedule(tmp_path)["rows"][0]
    assert row["per_sm"]["94"]["peak_overlapping_intervals"] == 2
