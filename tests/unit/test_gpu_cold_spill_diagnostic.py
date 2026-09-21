import copy

import pytest

from triton_viz.tools.gpu_cold_resource_diagnostic import SERVICE_FEATURES
from triton_viz.tools.gpu_spill_transfer_audit import FEATURES
from triton_viz.tools.gpu_cold_spill_diagnostic import request_prediction, validate
from triton_viz.tools.gpu_instruction_transfer_audit import scale


def test_region_candidate_preserves_all_rows_and_both_nested_exclusions():
    counters, latencies = data()
    for row in latencies:
        row["scalar_pricing_features"] = dict(sfu_warps=0, shuffle_steps=0)
        if row["case"]["kind"] != "geometry_dot":
            row["source_program_count"] = 1
            row["source_execution"] = dict(
                loop_trace=dict(
                    schema="triton-viz.gpu-source-loops.v1", complete=True, loops=[]
                ),
                dot_ancestry=[
                    dict(
                        seq=i,
                        program=[0],
                        ancestor_dot_seqs=[],
                        input_shapes=row["dot_shapes"][0],
                    )
                    for i in range(5)
                ],
            )
    instructions = copy.deepcopy(latencies)
    for row in instructions:
        value = scale(row)
        row["instructions_per_warp"] = dict(
            LDL=2, STL=1, executed=value + 10, issued=value + 10
        )
    first = validate(
        counters,
        latencies,
        region_instruction_rows=instructions,
        issued_dot=True,
        scalar_layout=True,
    )
    changed = copy.deepcopy(instructions)
    for row in changed:
        if row["case"]["cv_group"] == "0":
            row["instructions_per_warp"]["executed"] += 99999
    second = validate(
        counters,
        latencies,
        region_instruction_rows=changed,
        issued_dot=True,
        scalar_layout=True,
    )
    assert first["folds"][0] == second["folds"][0]
    predictions = first["ordinary"]["source_plus_region_instruction_work"]["rows"]
    assert len(predictions) == 8 and not any(r["source_fallback"] for r in predictions)
    assert first["eligible_for_fit"] is False
    issued = first["ordinary"]["source_plus_region_issued_dot_work"]["rows"]
    assert len(issued) == 8 and not any(r["source_fallback"] for r in issued)
    assert len(first["ordinary"]["source_plus_scalar_layout_work"]["rows"]) == 8
    for fold in first["folds"]:
        assert all(
            not key.startswith("l" + fold["held_group"] + "_")
            for key in fold["region_instruction_training_ids"]
        )


def test_instruction_candidate_excludes_both_nested_validation_layers():
    counters, latencies = data()
    instructions = copy.deepcopy(counters)
    for g, row in enumerate(instructions):
        row["instructions_per_warp"] = dict(
            LDL=10 * g,
            STL=5 * g,
            executed=scale(row) + 30 * g,
            issued=scale(row) + 30 * g,
        )
    first = validate(
        counters, latencies, include_regime=True, instruction_rows=instructions
    )
    changed = copy.deepcopy(instructions)
    changed[0]["instructions_per_warp"]["executed"] += 99999
    second = validate(
        counters, latencies, include_regime=True, instruction_rows=changed
    )
    assert first["folds"][0] == second["folds"][0]
    assert first["count"] == 8
    assert "source_plus_instruction_work" in first["ordinary"]
    base = {r["id"]: r for r in first["ordinary"]["source"]["rows"]}
    for row in first["ordinary"]["source_plus_instruction_work"]["rows"]:
        if row["source_fallback"]:
            assert row["prediction_us"] == base[row["id"]]["prediction_us"]
            assert "instruction_mapping_unmodeled_composition" in row["ood_reasons"]


def data():
    counters, latencies = [], []
    for g in range(4):
        source = dict(
            role="control",
            compiler_version="3.7.0",
            source_precision=[["fp16", "fp16", "ieee"]],
            dot_shapes=[[[32 * 2**g, 32], [32, 64]]],
            source_features={
                **dict.fromkeys(FEATURES, 1),
                "threads_per_program": 128,
                "dots_per_program": 5,
                "max_dot_m": 32 * 2**g,
            },
        )
        counters.append(
            dict(
                **source,
                case=dict(id=f"c{g}", cv_group=str(g), kind="geometry_dot"),
                counter_bytes_per_thread=dict(LDL=64 * (g + 1), STL=32 * (g + 1)),
            )
        )
        for kind in ("geometry_dot", "structure_composition"):
            latencies.append(
                dict(
                    **source,
                    case=dict(id=f"l{g}_{kind}", cv_group=str(g), kind=kind),
                    eligible_for_fit=False,
                    latency_us=float(10 + g),
                    pricing_features={
                        **dict.fromkeys(SERVICE_FEATURES, 0),
                        "launch": 1,
                        "global_sectors": 10 * (g + 1),
                        "waves": 1,
                    },
                )
            )
    return counters, latencies


@pytest.mark.parametrize("include_regime", [False, True])
def test_outer_counter_and_timing_labels_cannot_affect_selection(include_regime):
    counters, latencies = data()
    first = validate(counters, latencies, include_regime=include_regime)
    changed_c, changed_l = copy.deepcopy((counters, latencies))
    changed_c[0]["counter_bytes_per_thread"] = dict(LDL=999999, STL=999999)
    for row in changed_l:
        if row["case"]["cv_group"] == "0":
            row["latency_us"] = 999999
    second = validate(changed_c, changed_l, include_regime=include_regime)
    assert first["folds"][0] == second["folds"][0]
    for name in first["ordinary"]:
        assert [
            r["prediction_us"]
            for r in first["ordinary"][name]["rows"]
            if r["group"] == "0"
        ] == [
            r["prediction_us"]
            for r in second["ordinary"][name]["rows"]
            if r["group"] == "0"
        ]
    assert first["count"] == 8 and len(first["nested_rows"]) == 8


def test_compositions_retained_with_explicit_baseline_not_fake_requests():
    result = validate(*data())
    base = {r["id"]: r for r in result["ordinary"]["source"]["rows"]}
    fallback = [
        r
        for r in result["ordinary"]["source_plus_requests"]["rows"]
        if r["source_fallback"]
    ]
    assert len(fallback) == 4
    for row in fallback:
        assert row["prediction_us"] == base[row["id"]]["prediction_us"]
        assert "spill_mapping_unmodeled_composition" in row["ood_reasons"]
    assert result["eligible_for_fit"] is False and result["released_model"] is None


def test_holdout_and_shared_geometry_partition_leaks_rejected():
    counters, latencies = data()
    counters[0]["role"] = "holdout"
    with pytest.raises(ValueError, match="control"):
        validate(counters, latencies)
    counters[0]["role"] = "control"
    counters[0]["case"]["cv_group"] = "another"
    with pytest.raises(ValueError, match="partitions"):
        validate(counters, latencies)


def test_request_prediction_needs_no_query_counter_or_compiler_artifact():
    counters, _ = data()
    query = counters.pop()
    source_only = {
        k: query[k]
        for k in (
            "source_features",
            "source_precision",
            "dot_shapes",
            "compiler_version",
        )
    }
    first = request_prediction(counters, source_only)
    query["counter_bytes_per_thread"] = {"LDL": float("nan"), "STL": -1}
    query["local_bytes_per_thread"] = 99999999
    query["latency_us"] = 99999999
    assert request_prediction(counters, query) == first
    assert first["ood_reasons"]
