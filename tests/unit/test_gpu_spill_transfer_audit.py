import copy

import pytest

from triton_viz.tools.gpu_spill_transfer_audit import (
    FEATURES,
    dot_instructions,
    validate,
)


def controls():
    rows = []
    for g, m in enumerate((32, 64, 128, 256)):
        for stage in (1, 2):
            rows.append(
                dict(
                    role="control",
                    compiler_version="3.7.0",
                    case=dict(id=f"{g}_{stage}", cv_group=str(g)),
                    source_features={
                        **dict.fromkeys(FEATURES, 1),
                        "max_dot_m": m,
                        "threads_per_program": 128,
                        "dots_per_program": 5,
                        "requested_stages": stage,
                    },
                    source_precision=[["fp16", "fp16", "ieee"]],
                    dot_shapes=[[[m, 32], [32, 128]]],
                    source_program_count=48,
                    ood_reasons=[],
                    counter_bytes_per_thread=dict(LDL=g * 128, STL=g * 64),
                )
            )
    return rows


def test_pinned_source_instruction_units():
    row = controls()[0]
    assert dot_instructions(row) == 16
    row["source_precision"] = [["fp32", "fp32", "tf32"]]
    assert dot_instructions(row) == 32
    row["source_precision"] = [["fp32", "fp32", "ieee"]]
    assert dot_instructions(row) == 1024
    row["compiler_version"] = "unknown"
    with pytest.raises(ValueError):
        dot_instructions(row)


@pytest.mark.parametrize("normalization", ["dot", "instruction"])
def test_held_counter_labels_cannot_change_prediction(normalization):
    rows = controls()
    first = validate(rows, normalization=normalization)
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["case"]["cv_group"] == "0":
            row["counter_bytes_per_thread"] = dict(LDL=999999, STL=999999)
    second = validate(changed, normalization=normalization)
    assert [r["predicted"] for r in first["rows"][:2]] == [
        r["predicted"] for r in second["rows"][:2]
    ]
    assert first["count"] == 8 and first["released_model"] is None
    assert first["metrics"]["LDL"]["false_positive"] == 2
    for row in first["rows"]:
        assert row["case"]["id"] not in row["training_ids"]
        assert set(row["neighbors"]) <= set(row["training_ids"])


def test_all_zero_traffic_not_deleted_or_epsilon_divided():
    rows = controls()
    for row in rows:
        row["counter_bytes_per_thread"] = dict(LDL=0, STL=0)
    result = validate(rows, normalization="dot")
    assert result["count"] == 8
    assert result["metrics"]["LDL"]["payload_sector_wape_pct"] is None
    assert result["metrics"]["LDL"]["mae_bytes_per_thread"] == 0
    rows[0]["role"] = "holdout"
    with pytest.raises(ValueError, match="control"):
        validate(rows, normalization="dot")


def test_complete_existing_pure_dot_counter_phase(tmp_path):
    from microbench.gpu.common.cases import load_cases
    from triton_viz.tools.gpu_local_counter_collect import commands

    original = load_cases("resource_transfer", "control")
    phase = load_cases("resource_dot", "control")
    assert len(original) == 192 and len(phase) == 96
    assert phase == [r for r in original if r["kind"] == "geometry_dot"]
    assert load_cases("resource_dot", "holdout") == []
    generated = commands("ncu", tmp_path, suite="resource_dot")
    assert len(generated) == 96
    assert [c[c.index("--case-id") + 1] for c in generated] == [r["id"] for r in phase]
