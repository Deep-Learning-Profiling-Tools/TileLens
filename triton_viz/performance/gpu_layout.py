"""Source-only, versioned initial blocked-dot layout descriptors.

This describes Triton 3.7's initial rank-2 conversion policy, not all subsequent
layout optimizations or physical register allocation. Validate final layouts on
controls before using the descriptors in a calibrated prediction model.

Policy references (Triton v3.7.0): TritonToTritonGPUPass.cpp TritonDotPattern,
and TritonGPUAttrDefs.td BlockedEncodingAttr shape-based builder.
"""

import math


def initial_ieee_dot_layout(m, n, k, warps, *, compiler_version):
    """Describe one CTA of a power-of-two FP32 IEEE dot without compilation."""
    if compiler_version != "3.7.0":
        raise ValueError("Unvalidated compiler layout policy version")
    if any(
        isinstance(x, bool) or not isinstance(x, int) or x < 1 or x & (x - 1)
        for x in (m, n, k, warps)
    ) or warps not in (4, 8):
        raise ValueError("Require power-of-two shapes and control-covered warp count")
    density = m * n // (32 * warps)
    scalar_tile = 4 if density >= 16 else 2 if density >= 4 else 1
    size = [min(m, scalar_tile), min(n, scalar_tile)]
    # Fill the contiguous N axis first; remaining lanes/warps cover M.
    n_threads = min(32 * warps, max(1, n // size[1]))
    n_lanes = min(32, n_threads)
    n_warps = min(warps, max(1, n_threads // n_lanes))
    lanes = [32 // n_lanes, n_lanes]
    cta_warps = [warps // n_warps, n_warps]
    fragment = [
        math.ceil(dim / (s * lane * warp)) * s
        for dim, s, lane, warp in zip((m, n), size, lanes, cta_warps)
    ]
    return dict(
        size_per_thread=size,
        threads_per_warp=lanes,
        warps_per_cta=cta_warps,
        order=[1, 0],
        accumulator_shape=[m, n],
        k=k,
        accumulator_words_per_thread=math.prod(fragment),
        fully_materialized_operand_words_per_thread=k * sum(fragment),
    )
