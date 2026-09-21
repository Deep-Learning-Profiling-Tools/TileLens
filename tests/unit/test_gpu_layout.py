import pytest

from triton_viz.performance.gpu_layout import initial_ieee_dot_layout
from triton_viz.performance.gpu_layout import ieee_row_store_exchange


@pytest.mark.parametrize("dtype,total", [("tf32", 448), ("fp16", 352), ("bf16", 352)])
def test_mma_fragment_words_include_cross_warp_operand_replication(dtype, total):
    from triton_viz.performance.gpu_layout import mma_v2_fragments

    plan = mma_v2_fragments(
        256, 128, 32, 4, input_dtype=dtype, chained_dot=False, compiler_version="3.7.0"
    )
    assert plan["warp_layout"] == [2, 2]
    assert plan["accumulator_words_per_thread"] == 256
    assert plan["fully_materialized_words_per_thread"] == total
    bits = 32 if dtype == "tf32" else 16
    assert (
        plan["a_words_per_thread"] * 128
        == 256 * 32 * bits // 32 * plan["a_cross_warp_replication"]
    )
    assert (
        plan["b_words_per_thread"] * 128
        == 128 * 32 * bits // 32 * plan["b_cross_warp_replication"]
    )


def test_mma_chain_layout_changes_operands_without_changing_accumulator_demand():
    from triton_viz.performance.gpu_layout import mma_v2_fragments

    plain = mma_v2_fragments(
        64, 64, 32, 4, input_dtype="bf16", chained_dot=False, compiler_version="3.7.0"
    )
    chain = mma_v2_fragments(
        64, 64, 32, 4, input_dtype="bf16", chained_dot=True, compiler_version="3.7.0"
    )
    assert (
        plain["accumulator_words_per_thread"]
        == chain["accumulator_words_per_thread"]
        == 32
    )
    assert (plain["a_words_per_thread"], plain["b_words_per_thread"]) == (16, 16)
    assert (chain["a_words_per_thread"], chain["b_words_per_thread"]) == (8, 32)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(k=8),
        dict(k=True),
        dict(k=31),
        dict(k=32.0),
        dict(input_dtype="fp32"),
        dict(compiler_version="unknown"),
        dict(m=16, n=8),
    ],
)
def test_mma_fragments_reject_unverified_precision_tiles_and_policy(overrides):
    from triton_viz.performance.gpu_layout import mma_v2_fragments

    args = dict(
        m=64,
        n=64,
        k=32,
        warps=4,
        input_dtype="fp16",
        chained_dot=False,
        compiler_version="3.7.0",
    )
    with pytest.raises(ValueError):
        mma_v2_fragments(**(args | overrides))


@pytest.mark.parametrize(
    "m,warps,shuffles,barriers",
    [
        (32, 4, 10, 2),
        (32, 8, 11, 2),
        (64, 4, 4, 0),
        (64, 8, 4, 0),
    ],
)
def test_mma_row_reduction_separates_partial_reductions(m, warps, shuffles, barriers):
    from triton_viz.performance.gpu_layout import mma_v2_row_reduction

    plan = mma_v2_row_reduction(
        m, 64, warps, chained_dot=True, compiler_version="3.7.0"
    )
    assert plan["total_shuffles"] == shuffles
    assert plan["reduction_barriers"] == barriers


@pytest.mark.parametrize(
    "m,n,w,rounds",
    [(128, 256, 4, 32), (128, 256, 8, 8), (128, 64, 4, 4), (128, 64, 8, 2)],
)
def test_row_store_exchange_accounts_for_repeated_shared_tiles(m, n, w, rounds):
    result = ieee_row_store_exchange(
        m, n, w, compiler_version="3.7.0", alignment_bytes=16
    )
    assert result["kind"] == "shared" and result["shared_rounds"] == rounds
    assert result["conversion_barriers"] == 2 * rounds - 1
    assert result["shared_payload_bytes_per_program"] == 2 * m * n * 4


def test_warp_only_exchange_is_not_silently_priced_as_free():
    result = ieee_row_store_exchange(
        32, 64, 8, compiler_version="3.7.0", alignment_bytes=16
    )
    assert result["kind"] == "warp" and result["conversion_barriers"] is None
    assert result["reason"] == "warp_shuffle_or_shared_fallback_unmodeled"
    with pytest.raises(ValueError, match="alignment"):
        ieee_row_store_exchange(
            128, 256, 4, compiler_version="3.7.0", alignment_bytes=8
        )


def test_source_only_layout_reproduces_blocked_component_geometry():
    layout = initial_ieee_dot_layout(32, 64, 64, 4, compiler_version="3.7.0")
    assert layout["size_per_thread"] == [4, 4]
    assert layout["threads_per_warp"] == [2, 16]
    assert layout["warps_per_cta"] == [4, 1]
    assert layout["accumulator_words_per_thread"] == 16
    assert layout["fully_materialized_operand_words_per_thread"] == 512


def test_warp_count_changes_operand_replication_not_only_total_dot_work():
    four = initial_ieee_dot_layout(64, 64, 64, 4, compiler_version="3.7.0")
    eight = initial_ieee_dot_layout(64, 64, 64, 8, compiler_version="3.7.0")
    assert four["fully_materialized_operand_words_per_thread"] == 768
    assert eight["fully_materialized_operand_words_per_thread"] == 512


@pytest.mark.parametrize(
    "args", [(0, 64, 32, 4), (63, 64, 32, 4), (True, 64, 32, 4), (64, 64, 32, 3)]
)
def test_layout_rejects_unsupported_shapes(args):
    with pytest.raises(ValueError):
        initial_ieee_dot_layout(*args, compiler_version="3.7.0")


def test_layout_does_not_assume_compiler_policy_is_stable():
    with pytest.raises(ValueError, match="version"):
        initial_ieee_dot_layout(64, 64, 32, 4, compiler_version="unknown")


@pytest.mark.parametrize(
    "m,n,warps,single,chain",
    [
        (32, 64, 4, [1, 4], [1, 4]),
        (64, 64, 4, [2, 2], [4, 1]),
        (32, 64, 8, [2, 4], [1, 8]),
        (64, 64, 8, [2, 4], [8, 1]),
    ],
)
def test_mma_v2_connectivity_changes_warp_ownership(m, n, warps, single, chain):
    from triton_viz.performance.gpu_layout import mma_v2_warp_layout

    for connected, expected in ((False, single), (True, chain)):
        assert (
            mma_v2_warp_layout(
                m, n, warps, chained_dot=connected, compiler_version="3.7.0"
            )
            == expected
        )
    with pytest.raises(ValueError, match="connectivity"):
        mma_v2_warp_layout(m, n, warps, chained_dot=None, compiler_version="3.7.0")
