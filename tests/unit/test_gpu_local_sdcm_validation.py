import copy

import pytest

from triton_viz.tools.gpu_local_sdcm_validation import prediction, scenarios, validate


def controls():
    rows = []
    for slots in (32, 64, 128, 256):
        for programs in (48, 96, 192, 384):
            row = dict(
                local_slots=slots,
                programs=programs,
                sm_count=48,
                footprint_bytes=slots * programs * 128 * 4,
            )
            row.update(
                prediction(row, dict(capacity_bytes=128 * 1024, associativity=8))
            )
            rows.append(row)
    return rows


def test_synthetic_hypothesis_is_recoverable_without_publication():
    report = validate(controls())
    assert len(scenarios()) == 49
    assert report["count"] == 16
    assert report["full_control_score"]["load"]["mae_percentage_points"] == 0
    assert report["released_model"] is None
    assert report["eligible_for_fit"] is False
    for result in report["grouped_validation"].values():
        assert sum(len(f["predictions"]) for f in result["folds"]) == 16


def test_outer_labels_cannot_change_outer_or_inner_selection():
    rows = controls()
    before = validate(rows)["grouped_validation"]["local_slots"]["folds"][0]
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["local_slots"] == before["held_group"]:
            row["load"] = 0  # Keep exact zero labels; no epsilon or deletion.
            row["store"] = 0
    after = validate(changed)["grouped_validation"]["local_slots"]["folds"][0]
    assert before["selected"] == after["selected"]
    assert before["inner_folds"] == after["inner_folds"]
    assert before["inner_score"] == after["inner_score"]
    assert before["outer_score"] != after["outer_score"]


def test_prediction_does_not_read_labels():
    row = controls()[0]
    expected = prediction(row, scenarios()[0])
    del row["load"], row["store"]
    assert prediction(row, scenarios()[0]) == expected
    row["programs"] = 49
    with pytest.raises(ValueError):
        prediction(row, scenarios()[0])


def test_zero_miss_labels_are_retained_without_epsilon_percentage_error():
    rows = controls()
    for row in rows:
        row.update(load=1.0, store=1.0)
    result = validate(rows)
    assert result["count"] == 16
    for axis in result["grouped_validation"].values():
        for op in ("load", "store"):
            assert axis["outer_score"][op]["observed_miss_sectors"] == 0
            assert axis["outer_score"][op]["miss_count_wape_pct"] is None
            assert axis["outer_score"][op]["absolute_miss_sector_error"] > 0
