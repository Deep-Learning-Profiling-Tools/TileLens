import copy

import pytest

from triton_viz.performance.gpu_source_regions import (
    dot_execution_regions,
    conditional_mma_work,
)


def test_region_work_keeps_repeated_accumulator_separate_from_static_chain():
    dots = [dot(2), dot(12, [2])]
    args = dict(precision=[["bf16", "bf16", "ieee"]], warps=8, compiler_version="3.7.0")
    repeated = conditional_mma_work(
        dot_execution_regions(dots, trace([loop()])), **args
    )
    chained = conditional_mma_work(dot_execution_regions(dots, trace([])), **args)
    assert repeated["programs"][0]["instructions_per_warp"] == 16
    assert chained["programs"][0]["instructions_per_warp"] == 32
    assert not chained["compiler_regions_verified"]
    with pytest.raises(ValueError, match="precision"):
        conditional_mma_work(
            dot_execution_regions(dots, trace([])),
            **{**args, "precision": [["fp32", "fp32", "ieee"]]},
        )


def loop(start=0, end=20, depth=0, site=0, iterations=None):
    return dict(
        complete=True,
        program=[0],
        site=site,
        depth=depth,
        kind="python_range",
        event_start=start,
        event_end=end,
        iterations=iterations
        or [dict(event_start=0, event_end=10), dict(event_start=10, event_end=20)],
    )


def trace(loops):
    return dict(schema="triton-viz.gpu-source-loops.v1", complete=True, loops=loops)


def dot(seq, parents=()):
    return dict(
        seq=seq,
        program=[0],
        ancestor_dot_seqs=list(parents),
        input_shapes=[[64, 32], [32, 64]],
    )


def test_loop_carried_accumulator_is_not_same_iteration_dot_chain():
    result = dot_execution_regions([dot(2), dot(12, [2])], trace([loop()]))
    assert result["dot_count"] == 2
    assert result["same_iteration_edge_count"] == 0
    assert result["cross_region_edge_count"] == 1
    assert not result["compiler_regions_verified"]


def test_attention_like_pairs_preserve_within_iteration_and_cross_iteration_edges():
    dots = [dot(2), dot(4, [2]), dot(12), dot(14, [2, 4, 12])]
    result = dot_execution_regions(dots, trace([loop()]))
    assert [r["same_iteration_ancestry"] for r in result["regions"]] == [
        [[2, 4]],
        [[12, 14]],
    ]
    assert result["cross_region_edge_count"] == 2
    # Static-range lowering must not be silently inferred from dynamic paths.
    static = loop()
    static["kind"] = "static_range"
    assert (
        dot_execution_regions(dots, trace([static]))["same_iteration_edge_count"] == 2
    )


def test_straight_line_and_nested_regions_keep_every_dot():
    dots = [dot(0), dot(3, [0]), dot(7, [0, 3]), dot(15, [0, 3, 7])]
    nested = loop(
        start=1,
        end=9,
        depth=1,
        site=1,
        iterations=[dict(event_start=1, event_end=5), dict(event_start=5, event_end=9)],
    )
    result = dot_execution_regions(dots, trace([loop(), nested]))
    assert sum(len(r["dot_seqs"]) for r in result["regions"]) == 4
    assert result["same_iteration_edge_count"] == 0
    assert dot_execution_regions(dots, trace([]))["same_iteration_edge_count"] == 6


def test_invalid_loop_coverage_and_cross_program_ancestry_fail_closed():
    bad = loop()
    bad["iterations"][1]["event_start"] = 11
    with pytest.raises(ValueError, match="coverage"):
        dot_execution_regions([dot(2)], trace([bad]))
    missing = copy.deepcopy(trace([]))
    missing["complete"] = False
    with pytest.raises(ValueError, match="complete"):
        dot_execution_regions([], missing)
    other = dot(3, [2])
    other["program"] = [1]
    with pytest.raises(ValueError, match="cross-program"):
        dot_execution_regions([dot(2), other], trace([]))
