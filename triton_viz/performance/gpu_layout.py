"""Source-only, versioned dot layout descriptors.

This describes Triton 3.7's initial rank-2 conversion policy, not all subsequent
layout optimizations or physical register allocation. Validate final layouts on
controls before using the descriptors in a calibrated prediction model.

Policy references (Triton v3.7.0): TritonToTritonGPUPass.cpp TritonDotPattern,
TritonGPUAttrDefs.td BlockedEncodingAttr shape-based builder, and
AccelerateMatmul.cpp warpsPerTileV2. Connectivity and layout eligibility remain
explicit preconditions, not predictions from an operator name.
"""

import math


def mma_v2_warp_layout(m, n, warps, *, chained_dot, compiler_version):
    """Pinned rank-2 policy, conditional on a verified same-region dot chain.

    Mirrors warpsPerTileV2 in Triton 3.7 AccelerateMatmul.cpp. This is not
    MMA eligibility inference. Callers must establish the tensor path and
    connectivity; dynamic loop repetition alone is not a compiler dot chain.
    Existing layouts in heterogeneous chains require separate propagation.
    """
    if compiler_version != "3.7.0":
        raise ValueError("Unvalidated compiler layout policy version")
    if not isinstance(chained_dot, bool):
        raise ValueError("Require explicit verified dot connectivity")
    if any(
        isinstance(x, bool) or not isinstance(x, int) or x < 1 or x & (x - 1)
        for x in (m, n, warps)
    ):
        raise ValueError("Require power-of-two geometry and warp count")
    if warps not in (4, 8) or m < 16 or n < 8:
        raise ValueError("Outside validated rank-2 MMA policy domain")
    if chained_dot:
        return [warps, 1] if m >= n else [1, warps]
    reps = [(m + 15) // 16, (n + 7) // 8]
    layout = [1, 1]
    while math.prod(layout) < warps:
        axis = 0 if reps[0] >= reps[1] else 1
        layout[axis] *= 2
        if reps[axis] != 1:
            reps[axis] //= 2
    return layout


def mma_v2_issued_work(m, n, k, warps, *, input_dtype, chained_dot, compiler_version):
    """Conditional per-warp MMA expansion including replicated warp tiles.

    This counts emitted instruction slots, not useful arithmetic, active lanes,
    allocated registers or runtime. Connectivity and MMA-v2 eligibility are
    preconditions. Round-up captures warp layouts larger than logical geometry;
    control compiler/counter evidence must validate the physical execution.
    """
    if input_dtype not in {"fp16", "bf16", "tf32"}:
        raise ValueError("Require supported MMA precision")
    wm, wn = mma_v2_warp_layout(
        m, n, warps, chained_dot=chained_dot, compiler_version=compiler_version
    )
    ik = 8 if input_dtype == "tf32" else 16
    if isinstance(k, bool) or not isinstance(k, int) or k < ik or k % ik:
        raise ValueError("Require complete instruction K tiles")
    mt, nt = (m + 16 * wm - 1) // (16 * wm), (n + 8 * wn - 1) // (8 * wn)
    return dict(
        warp_layout=[wm, wn],
        instruction_shape=[16, 8, ik],
        instructions_per_warp=mt * nt * (k // ik),
        covered_shape=[16 * wm * mt, 8 * wn * nt, k],
        warp_layout_exceeds_geometry=(16 * wm > m or 8 * wn > n),
    )


def mma_v2_fragments(m, n, k, warps, *, input_dtype, chained_dot, compiler_version):
    """Fully materialized MMA-v2 fragment words, not physical register allocation.

    Conditional on the chosen tensor path and verified static-region layout.
    PTX m16n8k16 FP16/BF16 and m16n8k8 TF32 each use four A words,
    two B words and four FP32 accumulator words per instruction tile.
    A is replicated across N warps, B across M warps. Scheduling may shorten
    live ranges; these counts do not assert simultaneous liveness or spills.
    """
    if input_dtype not in {"fp16", "bf16", "tf32"}:
        raise ValueError("Require a supported MMA-v2 input precision")
    wm, wn = mma_v2_warp_layout(
        m, n, warps, chained_dot=chained_dot, compiler_version=compiler_version
    )
    instruction_k = 8 if input_dtype == "tf32" else 16
    if (
        isinstance(k, bool)
        or not isinstance(k, int)
        or k < instruction_k
        or k & (k - 1)
        or m < 16 * wm
        or n < 8 * wn
    ):
        raise ValueError(
            "Partial or broadcast instruction tiles require separate mapping"
        )
    mt, nt, kt = m // (16 * wm), n // (8 * wn), k // instruction_k
    a, b, acc = 4 * mt * kt, 2 * nt * kt, 4 * mt * nt
    return dict(
        warp_layout=[wm, wn],
        instruction_shape=[16, 8, instruction_k],
        a_words_per_thread=a,
        b_words_per_thread=b,
        accumulator_words_per_thread=acc,
        fully_materialized_words_per_thread=a + b + acc,
        a_cross_warp_replication=wn,
        b_cross_warp_replication=wm,
    )


def mma_v2_row_reduction(m, n, warps, *, chained_dot, compiler_version):
    """Static FP32 sum/max shuffle expansion for axis-1 MMA-v2 reduction.

    One scalar reduction, not normalization or layout-conversion costs. MMA-v2
    has two row registers and four column lanes per 16x8 instruction tile.
    Cross-warp partials use the scratch/second-shuffle scheme in Triton 3.7
    ReduceOpToLLVM. Counts are static per-thread instructions, not time.
    """
    wm, wn = mma_v2_warp_layout(
        m, n, warps, chained_dot=chained_dot, compiler_version=compiler_version
    )
    row_registers = 2 * math.ceil(m / (16 * wm))
    column_warps = min(wn, n // 8)
    within = row_registers * 2  # log2(4) column lanes
    cross = 0
    if column_warps > 1:
        scratch_rounds = max(m * column_warps // (32 * warps), 1)
        cross = scratch_rounds * (column_warps.bit_length() - 1)
    return dict(
        warp_layout=[wm, wn],
        within_warp_shuffles=within,
        partial_reduction_shuffles=cross,
        total_shuffles=within + cross,
        reduction_barriers=2 if column_warps > 1 else 0,
    )


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


def ieee_row_store_exchange(m, n, warps, *, compiler_version, alignment_bytes):
    """Conditional Level-A exchange plan for aligned contiguous FP32 row stores.

    Axis-aligned, non-broadcast blocked layouts only. Common register-coordinate
    bits become repetitions except the common contiguous 128-bit vector. This
    models the conversion itself, not surrounding scratch hazards or pipeline
    drains. Warp-only conversions need a separate shuffle/fallback model.
    """
    if alignment_bytes != 16 or isinstance(alignment_bytes, bool):
        raise ValueError("Require the validated 16-byte store alignment contract")
    src = initial_ieee_dot_layout(m, n, 1, warps, compiler_version=compiler_version)
    size = [1, min(n, 4)]
    n_threads = min(32 * warps, n // size[1])
    n_lanes = min(32, n_threads)
    n_warps = max(1, n_threads // n_lanes)
    dst = dict(
        size_per_thread=size,
        threads_per_warp=[32 // n_lanes, n_lanes],
        warps_per_cta=[warps // n_warps, n_warps],
        order=[1, 0],
    )

    def ownership(layout):
        registers, lanes, cta_warps = set(), [], []
        for axis in (1, 0):
            dim = (m, n)[axis]
            size_words, lane_count, warp_count = (
                layout[key][axis]
                for key in ("size_per_thread", "threads_per_warp", "warps_per_cta")
            )
            if size_words * lane_count * warp_count > dim:
                raise ValueError(
                    "Broadcast layouts require separate exchange accounting"
                )
            sb, lb, wb, db = (
                x.bit_length() - 1 for x in (size_words, lane_count, warp_count, dim)
            )
            registers.update((axis, b) for b in range(sb))
            lanes.extend((axis, b) for b in range(sb, sb + lb))
            cta_warps.extend((axis, b) for b in range(sb + lb, sb + lb + wb))
            registers.update((axis, b) for b in range(sb + lb + wb, db))
        return registers, lanes, cta_warps

    sr, sl, sw = ownership(src)
    dr, dl, dw = ownership(dst)
    base = dict(source_layout=src, destination_layout=dst)
    if sw == dw:
        same_thread = sl == dl
        return dict(
            **base,
            kind="thread" if same_thread else "warp",
            shared_rounds=0 if same_thread else None,
            conversion_barriers=0 if same_thread else None,
            reason=None if same_thread else "warp_shuffle_or_shared_fallback_unmodeled",
        )
    common = sr & dr
    contiguous_bits = 0
    while contiguous_bits < 2 and (1, contiguous_bits) in common:
        contiguous_bits += 1
    rounds = 2 ** (len(common) - contiguous_bits)
    return dict(
        **base,
        kind="shared",
        shared_rounds=rounds,
        conversion_barriers=2 * rounds - 1,
        vector_words=2**contiguous_bits,
        shared_payload_bytes_per_program=2 * m * n * 4,
        reason=None,
    )
