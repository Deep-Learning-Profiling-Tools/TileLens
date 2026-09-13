"""Holdout-only adapter for NKI's four noncausal unequal-V attention shapes.

Not the unmodified Tilebench flash_attention implementation. Fixed source tiles,
no autotuning or target-dependent configuration selection.
"""

import triton
import triton.language as tl


@triton.jit
def attention(Q, K, V, Out, DV: tl.constexpr):
    m = tl.program_id(0) * 32 + tl.arange(0, 32)
    d = tl.arange(0, 128)
    v = tl.program_id(1) * 64 + tl.arange(0, 64)
    n = tl.arange(0, 32)
    q = tl.load(Q + m[:, None] * 128 + d[None, :])
    maximum = tl.full((32,), -float("inf"), tl.float32)
    denominator = tl.zeros((32,), tl.float32)
    accumulator = tl.zeros((32, 64), tl.float32)
    for start in range(0, 128, 32):
        k = tl.load(K + (start + n[None, :]) * 128 + d[:, None])
        value = tl.load(V + (start + n[:, None]) * DV + v[None, :])
        scores = tl.dot(q, k, input_precision="ieee") * 0.08838834764831845
        new_maximum = tl.maximum(maximum, tl.max(scores, 1))
        correction = tl.exp(maximum - new_maximum)
        probability = tl.exp(scores - new_maximum[:, None])
        denominator = denominator * correction + tl.sum(probability, 1)
        accumulator = accumulator * correction[:, None] + tl.dot(
            probability, value, input_precision="ieee"
        )
        maximum = new_maximum
    tl.store(Out + m[:, None] * DV + v[None, :], accumulator / denominator[:, None])
