import pytest

from triton_viz.performance.gpu_layout import initial_ieee_dot_layout
from triton_viz.performance.gpu_layout import ieee_row_store_exchange


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
