import pytest

from triton_viz.performance.gpu_issue_features import dot_issue_features


def test_source_instruction_work_uses_warps_and_hardware_waves_only():
    args = dict(
        precision=[["bf16", "bf16", "ieee"]],
        instructions_per_warp=48,
        threads_per_program=256,
        waves=2,
    )
    result = dot_issue_features(**args)
    assert result["wave_dot_instructions_bf16"] == 768
    assert sum(result.values()) == 768
    assert (
        dot_issue_features(**{**args, "waves": 0.5})["wave_dot_instructions_bf16"]
        == 192
    )
    for bad in (True, 0, -1, float("nan")):
        with pytest.raises(ValueError):
            dot_issue_features(**{**args, "waves": bad})
    with pytest.raises(ValueError, match="precision"):
        dot_issue_features(**{**args, "precision": [["fp32", "fp32", "tf32x3"]]})
