"""Observed dot dependency regions from source loop events, without compilation.

An execution region is a program plus its observed nested iteration path. It is
not a claim about compiler IR regions, unrolling, branches or eventual layouts.
In particular, static-range iterations remain explicit until lowering policy
is independently established. Preserve cross-region ancestry instead of using
it as evidence that separate static dots form a compiler chain.
"""

from triton_viz.performance.gpu_layout import mma_v2_issued_work


def conditional_mma_work(regions, *, precision, warps, compiler_version):
    """Source-only region hypothesis; compiler labels are not accepted here.

    Eligibility is deliberately separate from observed ancestry. Heterogeneous
    connected output geometries require layout propagation not implemented here.
    Returned work must be control-validated before calibrating a predictor.
    """
    signatures = {
        (("fp32", "fp32", "tf32"),): "tf32",
        (("bf16", "bf16", "ieee"),): "bf16",
        (("fp16", "fp16", "ieee"),): "fp16",
    }
    dtype = signatures.get(tuple(tuple(p) for p in precision))
    if dtype is None:
        raise ValueError("Outside homogeneous MMA-v2 precision scope")
    plans = []
    by_program = {}
    for region in regions["regions"]:
        shapes = dict(zip(region["dot_seqs"], region["dot_shapes"]))
        for a, b in shapes.values():
            if len(a) != 2 or len(b) != 2 or a[1] != b[0]:
                raise ValueError("Invalid rank-two source dot geometry")
        connected = set()
        for parent, child in region["same_iteration_ancestry"]:
            a, b = shapes[parent], shapes[child]
            if (a[0][0], a[1][1]) != (b[0][0], b[1][1]):
                raise ValueError("Heterogeneous chain requires layout propagation")
            connected.update((parent, child))
        for seq, (a, b) in shapes.items():
            plan = mma_v2_issued_work(
                a[0],
                b[1],
                a[1],
                warps,
                input_dtype=dtype,
                chained_dot=seq in connected,
                compiler_version=compiler_version,
            )
            program = tuple(region["program"])
            by_program[program] = (
                by_program.get(program, 0) + plan["instructions_per_warp"]
            )
            plans.append(
                dict(seq=seq, program=list(program), chained=seq in connected, **plan)
            )
    return dict(
        dot_plans=plans,
        programs=[
            dict(program=list(p), instructions_per_warp=v)
            for p, v in by_program.items()
        ],
        compiler_regions_verified=False,
        caveat="Conditional on observed iteration boundaries matching MMA layout-selection regions; not a verified arbitrary-kernel lowering rule.",
    )


def dot_execution_regions(dot_ancestry, loop_trace):
    if (
        loop_trace.get("schema") != "triton-viz.gpu-source-loops.v1"
        or loop_trace.get("complete") is not True
    ):
        raise ValueError("Require complete source loop observation")
    loops = {}
    for occurrence, loop in enumerate(loop_trace["loops"]):
        start, end = loop["event_start"], loop["event_end"]
        if (
            loop.get("complete") is not True
            or any(
                isinstance(v, bool) or not isinstance(v, int) or v < 0
                for v in (start, end, loop["depth"], loop["site"])
            )
            or end < start
        ):
            raise ValueError("Invalid loop extent")
        cursor = start
        for iteration in loop["iterations"]:
            a, b = iteration["event_start"], iteration["event_end"]
            if (
                any(isinstance(v, bool) or not isinstance(v, int) for v in (a, b))
                or a != cursor
                or not a <= b <= end
            ):
                raise ValueError("Invalid iteration coverage")
            cursor = b
        if cursor != end:
            raise ValueError("Incomplete iteration coverage")
        loops.setdefault(tuple(loop["program"]), []).append((occurrence, loop))
    known, grouped = {}, {}
    previous = -1
    for dot in dot_ancestry:
        seq = dot["seq"]
        program = tuple(dot["program"])
        if isinstance(seq, bool) or not isinstance(seq, int) or seq <= previous:
            raise ValueError("Require increasing unique dot sequence IDs")
        previous = seq
        parents = dot["ancestor_dot_seqs"]
        if len(set(parents)) != len(parents) or any(
            p not in known or known[p] != program for p in parents
        ):
            raise ValueError("Invalid or cross-program dot ancestry")
        known[seq] = program
        path = []
        for occurrence, loop in loops.get(program, []):
            if loop["event_start"] <= seq < loop["event_end"]:
                matches = [
                    i
                    for i, it in enumerate(loop["iterations"])
                    if it["event_start"] <= seq < it["event_end"]
                ]
                if len(matches) != 1:
                    raise ValueError("Dot outside unique loop iteration")
                path.append(
                    (loop["depth"], occurrence, loop["site"], matches[0], loop["kind"])
                )
        path.sort()
        if [p[0] for p in path] != list(range(len(path))):
            raise ValueError("Ambiguous or incomplete nested loop path")
        key = (program, tuple(path))
        grouped.setdefault(key, []).append(dot)
    regions = []
    for (program, path), dots in grouped.items():
        local = {d["seq"] for d in dots}
        edges = []
        cross = []
        for dot in dots:
            for parent in dot["ancestor_dot_seqs"]:
                (edges if parent in local else cross).append([parent, dot["seq"]])
        regions.append(
            dict(
                program=list(program),
                iteration_path=[
                    dict(depth=d, occurrence=o, site=s, iteration=i, kind=k)
                    for d, o, s, i, k in path
                ],
                dot_seqs=[d["seq"] for d in dots],
                same_iteration_ancestry=edges,
                cross_region_ancestry=cross,
                dot_shapes=[d["input_shapes"][:2] for d in dots],
            )
        )
    return dict(
        schema="triton-viz.gpu-dot-execution-regions.v1",
        regions=regions,
        dot_count=len(dot_ancestry),
        same_iteration_edge_count=sum(
            len(r["same_iteration_ancestry"]) for r in regions
        ),
        cross_region_edge_count=sum(len(r["cross_region_ancestry"]) for r in regions),
        compiler_regions_verified=False,
    )
