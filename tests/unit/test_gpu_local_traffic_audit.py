import hashlib

import pytest

from triton_viz.tools.gpu_local_traffic_audit import account


SASS = """
/*0010*/ STL.64 [R1], R2;
/*0020*/ LDL.LU.64 R2, [R1];
/*0030*/ STL [R1], R2;
/*0040*/ BRA.U UP0, 0x20;
/*0050*/ LDL R2, [R1];
/*0060*/ BRA 0x60;
"""


def row(sass=SASS):
    return dict(
        role="control",
        artifacts=dict(sass=sass),
        artifact_sha256=dict(sass=hashlib.sha256(sass.encode()).hexdigest()),
        case=dict(programs=48, num_warps=4),
    )


def test_widths_and_loop_trips_determine_conditional_payload():
    result = account(row(), loop_trips=5)
    assert result["static_bytes_per_thread"] == {
        "loop": {"LDL": 8, "STL": 4},
        "outside": {"LDL": 4, "STL": 8},
    }
    assert result["conditional_bytes_per_thread"] == {"LDL": 44, "STL": 28}
    assert result["conditional_payload_sector_equivalents"] == {
        "LDL": 44 * 48 * 4,
        "STL": 28 * 48 * 4,
    }
    assert not result["eligible_for_fit"]


def test_predicate_bounds_separate_loop_and_outside_without_exact_value_claim():
    sass = SASS.replace("LDL.LU.64", "@!P0 LDL.LU.64").replace("STL.64", "@UP1 STL.64")
    result = account(row(sass), loop_trips=5, predicate_bounds=True)
    assert result["conditional_bytes_per_thread_bounds"] == {
        "LDL": [4, 44],
        "STL": [20, 28],
    }
    assert result["conditional_payload_sector_equivalent_bounds"] == {
        "LDL": [4 * 192, 44 * 192],
        "STL": [20 * 192, 28 * 192],
    }
    assert "conditional_bytes_per_thread" not in result
    assert "conditional_payload_sector_equivalents" not in result
    assert not result["eligible_for_fit"]
    with pytest.raises(ValueError, match="active-lane"):
        account(row(sass), loop_trips=5)


def test_no_predicate_bounds_collapse_to_existing_conditional_accounting():
    exact = account(row(), loop_trips=5)
    bounds = account(row(), loop_trips=5, predicate_bounds=True)
    for op, value in exact["conditional_payload_sector_equivalents"].items():
        assert bounds["conditional_payload_sector_equivalent_bounds"][op] == [
            value,
            value,
        ]


def test_straight_line_unrolled_code_counts_once():
    r = row(SASS.replace("BRA.U UP0, 0x20", "BRA 0x40"))
    result = account(r, loop_trips=1)
    assert result["loop_region"] is None
    assert result["conditional_bytes_per_thread"] == {"LDL": 12, "STL": 12}
    with pytest.raises(ValueError, match="executes once"):
        account(r, loop_trips=5)


@pytest.mark.parametrize(
    "sass",
    [
        SASS.replace("LDL.LU.64", "@P0 LDL.LU.64"),
        SASS.replace("LDL.LU.64", "LDL.U8"),
        SASS.replace("/*0050*/ LDL R2, [R1];", "/*0050*/ BRA 0x70;"),
        SASS.replace("/*0050*/ LDL R2, [R1];", "/*0050*/ CALL 0x70;"),
        SASS.replace("0x20", "0x40"),
        SASS + "\n/*0050*/ LDL R2, [R1];",
    ],
)
def test_unsupported_paths_do_not_silently_price(sass):
    with pytest.raises(ValueError):
        account(row(sass), loop_trips=5)
    if "@P0" not in sass:
        with pytest.raises(ValueError):
            account(row(sass), loop_trips=5, predicate_bounds=True)


@pytest.mark.parametrize("predicate_bounds", [False, True])
def test_rejects_target_artifact_and_changed_digest(predicate_bounds):
    r = row()
    r["role"] = "holdout"
    with pytest.raises(ValueError, match="Only control"):
        account(r, loop_trips=5, predicate_bounds=predicate_bounds)
    r = row()
    r["artifact_sha256"]["sass"] = "wrong"
    with pytest.raises(ValueError, match="fingerprint"):
        account(r, loop_trips=5, predicate_bounds=predicate_bounds)


@pytest.mark.parametrize("trips", [True, 0, -1, 1.5])
def test_invalid_trip_hypotheses(trips):
    with pytest.raises(ValueError):
        account(row(), loop_trips=trips)
