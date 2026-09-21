import copy

import pytest

from triton_viz.tools.gpu_instruction_transfer_audit import (
    LABELS,
    predict,
    validate,
    decompose_work,
    predict_work,
    scale,
)
from triton_viz.tools.gpu_spill_transfer_audit import FEATURES


def test_execution_scale_accounts_for_chained_warp_replication_without_labels():
    row = controls()[0]
    row.update(source_program_count=1, source_precision=[["bf16", "bf16", "ieee"]])
    row["source_features"]["threads_per_program"] = 256
    row["source_features"]["dots_per_program"] = 2
    row["source_execution"] = dict(
        loop_trace=dict(
            schema="triton-viz.gpu-source-loops.v1", complete=True, loops=[]
        ),
        dot_ancestry=[
            dict(
                seq=0,
                program=[0],
                ancestor_dot_seqs=[],
                input_shapes=[[64, 32], [32, 64]],
            ),
            dict(
                seq=1,
                program=[0],
                ancestor_dot_seqs=[0],
                input_shapes=[[64, 64], [64, 64]],
            ),
        ],
    )
    assert scale(row) == 48
    clean = {
        k: v
        for k, v in row.items()
        if k not in {"case", "role", "instructions_per_warp"}
    }
    assert scale(clean) == 48
    training = work_controls()[1:]
    for control in training:
        control["source_precision"] = [["bf16", "bf16", "ieee"]]
    first = predict_work(training, clean)
    row["instructions_per_warp"] = dict.fromkeys(LABELS, 999999)
    assert predict_work(training, row) == first
    assert "instruction_region_lowering_conditional" in first["ood_reasons"]
    row["source_program_count"] = 2
    with pytest.raises(ValueError, match="complete uniform"):
        scale(row)
    clean["source_precision"] = [["fp32", "fp32", "ieee"]]
    assert scale(clean) == (64 * 32 * 64 + 64 * 64 * 64) // 256


def work_controls():
    rows = controls()
    for g, row in enumerate(rows):
        executed = scale(row) + 10 * g + 5 * g + 20 * g
        row["instructions_per_warp"] = dict(
            LDL=10 * g, STL=5 * g, executed=executed, issued=executed
        )
    return rows


def test_work_decomposition_conserves_instructions_and_rejects_negative_residual():
    rows = work_controls()
    for row in rows:
        assert (
            sum(decompose_work(row).values()) + scale(row)
            == row["instructions_per_warp"]["executed"]
        )
    rows[0]["instructions_per_warp"]["executed"] = 0
    with pytest.raises(ValueError, match="negative residual"):
        decompose_work(rows[0])


def test_work_prediction_does_not_decompose_query_labels():
    rows = work_controls()
    first = validate(rows, components=True)
    changed = copy.deepcopy(rows)
    changed[0]["instructions_per_warp"]["executed"] += 99999
    assert (
        first["rows"][0]["instructions_per_warp"]
        == validate(changed, components=True)["rows"][0]["instructions_per_warp"]
    )
    source = {
        k: rows[0][k]
        for k in (
            "compiler_version",
            "dot_shapes",
            "source_precision",
            "source_features",
        )
    }
    assert predict_work(rows[1:], source) == predict_work(rows[1:], changed[0])


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
