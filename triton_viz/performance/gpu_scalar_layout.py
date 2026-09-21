"""Conditional source mapping for MMA row reductions and normalization SFUs.

The source graph must expose reduction axes and ordered scalar operands.
Instruction demand is not a latency prediction or arbitrary lowering proof.
"""

import math

from triton_viz.performance.gpu_layout import mma_v2_row_reduction
from triton_viz.performance.gpu_resources import source_dot_ancestry
from triton_viz.performance.gpu_source_regions import (
    conditional_mma_work,
    dot_execution_regions,
)


def scalar_layout_work(source, *, precision, compiler_version):
    dots = source_dot_ancestry(source)
    regions = dot_execution_regions(dots, source["loop_trace"])
    warps = source["num_warps"]
    work = conditional_mma_work(
        regions, precision=precision, warps=warps, compiler_version=compiler_version
    )
    plans = {p["seq"]: p for p in work["dot_plans"]}
    shapes = {
        d["seq"]: (d["input_shapes"][0][0], d["input_shapes"][1][1]) for d in dots
    }
    nodes, frontier, rows = {}, {}, []

    def parent_plan(event, parents):
        signatures = {
            (shapes[p], tuple(plans[p]["warp_layout"]), plans[p]["chained"])
            for p in parents
        }
        if len(signatures) != 1:
            raise ValueError("Ambiguous source reduction/elementwise layout")
        (m, n), (wm, wn), chained = next(iter(signatures))
        return m, n, wm, wn, chained

    for event in source["events"]:
        seq, op = event["seq"], event["op"]
        parents = set().union(*(frontier[d] for d in event["dependencies"]))
        if op in {"trans", "transpose"}:
            parents.clear()
        if op == "dot":
            parents = {seq}
        elif parents and (
            op.startswith("reduce_") or event.get("primitive") in {"exp", "divide"}
        ):
            if event["dtype"] != "fp32":
                raise ValueError("Outside FP32 scalar lowering scope")
            m, n, wm, wn, chained = parent_plan(event, parents)
            fragment = 4 * math.ceil(m / (16 * wm)) * math.ceil(n / (8 * wn))
            plan = dict(
                seq=seq,
                program=event["program"],
                op=op,
                primitive=event.get("primitive"),
                sfu_per_warp=0,
                shuffles_per_warp=0,
                replaced_logical_sfu_warps=0,
                replaced_logical_shuffle_steps=0,
            )
            if op.startswith("reduce_"):
                if (
                    op not in {"reduce_max", "reduce_sum"}
                    or event.get("reduction_axis") not in {1, -1}
                    or event["input_shapes"] != [[m, n]]
                ):
                    raise ValueError("Require observed axis-1 MMA sum/max reduction")
                reduction = mma_v2_row_reduction(
                    m, n, warps, chained_dot=chained, compiler_version=compiler_version
                )
                plan["shuffles_per_warp"] = reduction["total_shuffles"]
                plan["replaced_logical_shuffle_steps"] = event["elements"] * math.ceil(
                    math.log2(n)
                )
            elif event.get("primitive") == "exp":
                if event["shape"] != [m, n]:
                    raise ValueError("Require full MMA accumulator exp")
                plan["sfu_per_warp"] = fragment
                plan["replaced_logical_sfu_warps"] = math.ceil(event["elements"] / 32)
            else:
                operands = event.get("operand_dependencies")
                if (
                    not operands
                    or len(operands) != 2
                    or operands[1] is None
                    or event["shape"] != [m, n]
                ):
                    raise ValueError("Require ordered division operands")
                denominator = nodes[operands[1]]
                for expected_op, expected_shape in (
                    ("broadcast", [m, n]),
                    ("expand_dims", [m, 1]),
                ):
                    if (
                        denominator["op"] != expected_op
                        or denominator["shape"] != expected_shape
                        or len(denominator["dependencies"]) != 1
                    ):
                        raise ValueError("Require explicit row-sum broadcast divisor")
                    denominator = nodes[denominator["dependencies"][0]]
                if (
                    denominator["op"] != "reduce_sum"
                    or denominator.get("reduction_axis") not in {1, -1}
                    or denominator["input_shapes"] != [[m, n]]
                ):
                    raise ValueError("Require explicit row-sum broadcast divisor")
                # Reciprocal is common to the row, not one per output element.
                plan["sfu_per_warp"] = 2 * math.ceil(m / (16 * wm))
            rows.append(plan)
        frontier[seq] = parents
        nodes[seq] = event
    return dict(
        rows=rows,
        sfu_warp_instructions=sum(r["sfu_per_warp"] * warps for r in rows),
        shuffle_warp_instructions=sum(r["shuffles_per_warp"] * warps for r in rows),
        replaced_logical_sfu_warps=sum(r["replaced_logical_sfu_warps"] for r in rows),
        replaced_logical_shuffle_steps=sum(
            r["replaced_logical_shuffle_steps"] for r in rows
        ),
        compiler_regions_verified=False,
        caveat="Conditional MMA exp, row-sum reciprocal and axis-1 reduction demand. No scratch-hazard/barrier, full division sequence, register allocation or latency claim.",
    )
