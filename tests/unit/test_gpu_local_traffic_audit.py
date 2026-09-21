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


def test_rejects_target_artifact_and_changed_digest():
    r = row()
    r["role"] = "holdout"
    with pytest.raises(ValueError, match="Only control"):
        account(r, loop_trips=5)
    r = row()
    r["artifact_sha256"]["sass"] = "wrong"
    with pytest.raises(ValueError, match="fingerprint"):
        account(r, loop_trips=5)


@pytest.mark.parametrize("trips", [True, 0, -1, 1.5])
def test_invalid_trip_hypotheses(trips):
    with pytest.raises(ValueError):
        account(row(), loop_trips=trips)
