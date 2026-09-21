"""Source-level resource-demand descriptors, not measured register allocation.

These features deliberately retain a distinction between logical values and
physical registers. Control compiler/counter data must establish that mapping;
source arithmetic alone does not reveal ptxas spills or resident CTA counts.
"""

from __future__ import annotations

import math


def source_dot_ancestry(source):
    """Export observed producer-dot ancestry, not compiler region inference.

    Transposes break the path as in the pinned MMA slice filter. Dynamic loop
    iterations can still appear connected: these descriptors do not establish
    distinct static dots or common compiler regions. Preserve sequence IDs so
    control audits can verify those additional conditions independently.
    """
    ancestry, programs, rows = {}, {}, []
    previous = None
    for event in source["events"]:
        seq = event["seq"]
        if previous is not None and seq <= previous:
            raise ValueError("Require increasing unique source sequence IDs")
        previous = seq
        parents = set()
        for dep in event["dependencies"]:
            if dep not in ancestry or programs[dep] != event["program"]:
                raise ValueError("Invalid or cross-program source dependency")
            parents.update(ancestry[dep])
        if event["op"] in {"trans", "transpose"}:
            parents.clear()
        if event["op"] == "dot":
            rows.append(
                dict(
                    seq=seq,
                    program=event["program"],
                    ancestor_dot_seqs=sorted(parents),
                    input_shapes=event["input_shapes"],
                )
            )
            parents.add(seq)
        ancestry[seq] = parents
        programs[seq] = event["program"]
    return rows


def source_liveness_features(source):
    """Measure logical SSA output lifetimes in the observed program order.

    This counts tensor values, including view-like outputs, not allocated
    registers. Compiler fusion, aliasing, rematerialization and pipelining can
    all change physical demand. No compiled artifact is consulted.
    """
    warps = source["num_warps"]
    if isinstance(warps, bool) or not isinstance(warps, int) or warps <= 0:
        raise ValueError("A positive source warp count is required")
    events = source["events"]
    by_seq = {}
    last_use = {}
    previous_seq = None
    for event in events:
        seq = event["seq"]
        if previous_seq is not None and seq <= previous_seq:
            raise ValueError("Source events must have unique increasing sequence IDs")
        previous_seq = seq
        for dep in event["dependencies"]:
            if dep not in by_seq or by_seq[dep]["program"] != event["program"]:
                raise ValueError("Invalid or cross-program source dependency")
            last_use[dep] = seq
        by_seq[seq] = event
        last_use[seq] = seq
    widths = {"fp32": 4, "fp16": 2, "bf16": 2}
    releases = {}
    live = peak = 0.0
    for event in events:
        words = event["elements"] * widths.get(event["dtype"], 0) / 4
        if not math.isfinite(words) or words < 0:
            raise ValueError("Invalid logical tensor size")
        live += words
        # Output and its last-used inputs overlap at the operation boundary.
        peak = max(peak, live)
        end = last_use[event["seq"]]
        releases[end] = releases.get(end, 0) + words
        live -= releases.pop(event["seq"], 0)
    return {"logical_live_float_words_per_thread": peak / (32 * warps)}


def source_resource_features(source):
    warps = source["num_warps"]
    stages = source["num_stages"]
    if isinstance(warps, bool) or not isinstance(warps, int) or warps <= 0:
        raise ValueError("A positive source warp count is required")
    if isinstance(stages, bool) or not isinstance(stages, int) or stages <= 0:
        raise ValueError("A positive source stage count is required")
    threads = 32 * warps
    accumulator_words = operand_words = float_tile_words = 0.0
    precision = set()
    reasons = set()
    widths = {"fp32": 4, "fp16": 2, "bf16": 2}
    for event in source["events"]:
        if event["dtype"] in widths:
            float_tile_words = max(
                float_tile_words, event["elements"] * widths[event["dtype"]] / 4
            )
        if event["op"] != "dot":
            continue
        shapes = event["input_shapes"]
        dtypes = event.get("dot_input_dtypes", [])
        if (
            len(shapes) < 2
            or any(len(shape) != 2 for shape in shapes[:2])
            or shapes[0][1] != shapes[1][0]
        ):
            reasons.add("unsupported_resource_dot_geometry")
            continue
        if (
            len(dtypes) != 2
            or any(dtype not in widths for dtype in dtypes)
            or event.get("dot_accumulator_dtype") != "fp32"
        ):
            reasons.add("unsupported_resource_dot_precision")
            continue
        if event.get("dot_input_precision") not in {"ieee", "tf32"}:
            reasons.add("unsupported_resource_dot_precision")
            continue
        precision.add(tuple([*dtypes, event["dot_input_precision"]]))
        accumulator_words = max(accumulator_words, shapes[0][0] * shapes[1][1])
        operand_words = max(
            operand_words,
            sum(
                math.prod(shape) * widths[dtype] / 4
                for shape, dtype in zip(shapes[:2], dtypes)
            ),
        )
    return (
        dict(
            logical_dot_accumulator_words_per_thread=accumulator_words / threads,
            logical_dot_operand_words_per_thread=operand_words / threads,
            max_float_tile_words_per_thread=float_tile_words / threads,
            requested_stages=stages,
            threads_per_program=threads,
        ),
        [list(value) for value in sorted(precision)],
        sorted(reasons),
    )
