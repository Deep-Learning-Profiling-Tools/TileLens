import pytest

from triton_viz.performance.gpu_layout import initial_ieee_dot_layout


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
