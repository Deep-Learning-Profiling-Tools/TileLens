import pytest

from triton_viz.performance.gpu_resources import source_resource_features


def source(*, m=256, n=128, k=32, warps=4, dtype="fp16"):
    return dict(
        num_warps=warps,
        num_stages=1,
        events=[
            dict(
                op="dot",
                dtype="fp32",
                elements=m * n,
                input_shapes=[[m, k], [k, n]],
                dot_input_dtypes=[dtype, dtype],
                dot_accumulator_dtype="fp32",
                dot_input_precision="ieee",
            )
        ],
    )


def test_logical_pressure_separates_geometry_precision_and_warp_count():
    fp16, _, reasons = source_resource_features(source())
    assert not reasons
    assert fp16["logical_dot_accumulator_words_per_thread"] == 256
    assert fp16["logical_dot_operand_words_per_thread"] == 48
    fp32, _, _ = source_resource_features(source(dtype="fp32"))
    assert fp32["logical_dot_operand_words_per_thread"] == 96
    wider, _, _ = source_resource_features(source(warps=8))
    assert wider["logical_dot_accumulator_words_per_thread"] == 128
    assert wider["logical_dot_operand_words_per_thread"] == 24
    repeated = source()
    repeated["events"] *= 5
    assert source_resource_features(repeated)[0] == fp16
    assert "registers_per_thread" not in fp16 and "spill_bytes" not in fp16


def test_unknown_resource_precision_is_explicit_not_guessed():
    work, _, reasons = source_resource_features(source(dtype="fp8"))
    assert reasons == ["unsupported_resource_dot_precision"]
    assert work["logical_dot_accumulator_words_per_thread"] == 0


@pytest.mark.parametrize("warps", [0, -1, True, 1.5])
def test_invalid_resource_launch_fails(warps):
    with pytest.raises(ValueError):
        source_resource_features(source(warps=warps))
