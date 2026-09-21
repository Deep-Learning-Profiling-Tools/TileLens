import copy

import pytest

from triton_viz.tools.gpu_instruction_transfer_audit import LABELS, predict, validate
from triton_viz.tools.gpu_spill_transfer_audit import FEATURES


def controls():
    return [
        dict(
            role="control",
            case=dict(id=str(g), cv_group=str(g)),
            compiler_version="3.7.0",
            dot_shapes=[[[32 * 2**g, 32], [32, 64]]],
            source_program_count=48,
            source_precision=[["fp16", "fp16", "ieee"]],
            source_features={
                **dict.fromkeys(FEATURES, 1),
                "max_dot_m": 32 * 2**g,
                "threads_per_program": 128,
                "dots_per_program": 5,
            },
            instructions_per_warp=dict.fromkeys(LABELS, 100 * g),
        )
        for g in range(4)
    ]


def test_held_instruction_labels_do_not_change_prediction():
    rows = controls()
    first = validate(rows)
    changed = copy.deepcopy(rows)
    changed[0]["instructions_per_warp"] = dict.fromkeys(LABELS, 999999)
    second = validate(changed)
    assert (
        first["rows"][0]["instructions_per_warp"]
        == second["rows"][0]["instructions_per_warp"]
    )
    assert first["count"] == 4 and first["released_model"] is None
    assert first["rows"][0]["training_ids"] == ["1", "2", "3"]


def test_query_needs_only_source_metadata_and_zero_labels_are_retained():
    rows = controls()
    query = rows.pop()
    clean = {
        k: query[k]
        for k in (
            "compiler_version",
            "dot_shapes",
            "source_precision",
            "source_features",
        )
    }
    assert predict(rows, clean) == predict(rows, query)
    for row in rows:
        row["instructions_per_warp"] = dict.fromkeys(LABELS, 0)
    result = validate(rows)
    assert result["count"] == 3
    assert result["metrics"]["LDL"]["instruction_count_wape_pct"] is None
    assert predict(rows, clean)["ood_reasons"]


def test_holdout_and_partition_leak_are_rejected():
    rows = controls()
    rows[0]["role"] = "holdout"
    with pytest.raises(ValueError, match="control"):
        validate(rows)
    rows = controls()
    rows[1]["dot_shapes"] = rows[0]["dot_shapes"]
    with pytest.raises(ValueError, match="partitions"):
        validate(rows)
