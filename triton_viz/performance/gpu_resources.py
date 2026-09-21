"""Source-level resource-demand descriptors, not measured register allocation.

These features deliberately retain a distinction between logical values and
physical registers. Control compiler/counter data must establish that mapping;
source arithmetic alone does not reveal ptxas spills or resident CTA counts.
"""

from __future__ import annotations

import math


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
