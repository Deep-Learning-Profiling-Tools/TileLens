"""Conditional source issue work for control-validated precision pathways.

These descriptors are instruction demand, not timing constants, allocated
registers, active-lane predictions or proof of arbitrary compiler eligibility.
Prediction reads source metadata only. The calibration audit owns admission.
"""

import math

ISSUE_DOT_FEATURES = tuple(
    "wave_dot_instructions_" + kind for kind in ("ieee_fp32", "tf32", "bf16", "fp16")
)


def dot_issue_features(*, precision, instructions_per_warp, threads_per_program, waves):
    """Convert per-warp issued work to SM-wave demand without hand constants."""
    kinds = {
        (("fp32", "fp32", "ieee"),): "ieee_fp32",
        (("fp32", "fp32", "tf32"),): "tf32",
        (("bf16", "bf16", "ieee"),): "bf16",
        (("fp16", "fp16", "ieee"),): "fp16",
    }
    kind = kinds.get(tuple(tuple(p) for p in precision))
    if kind is None:
        raise ValueError("Require homogeneous verified dot precision")
    if (
        isinstance(threads_per_program, bool)
        or not isinstance(threads_per_program, int)
        or threads_per_program <= 0
        or threads_per_program % 32
        or any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or v <= 0
            for v in (instructions_per_warp, waves)
        )
    ):
        raise ValueError("Invalid source issue work or hardware waves")
    result = dict.fromkeys(ISSUE_DOT_FEATURES, 0.0)
    result["wave_dot_instructions_" + kind] = (
        instructions_per_warp * (threads_per_program // 32) * waves
    )
    return result
