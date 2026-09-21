"""Straight-line shared-memory hazard accounting for verified source plans.

Input byte intervals must come from a separately validated source allocation
policy. This engine neither infers allocation nor accepts compiler artifacts.
It models conservative CTA RAW/WAR/WAW synchronization, not bank conflicts,
barrier duration, async pipelines, arbitrary control flow or lane ownership.
Those mechanisms need separate models; unsupported operations fail explicitly.
"""


def shared_dependency_barriers(operations):
    """Track pending accesses across explicit barriers and scratch operations.

    Scratch operations write then read, with internal CTA or warp synchronization.
    A warp-local barrier must not clear other warps' outstanding accesses.
    This follows the state transitions in Triton 3.7 MembarAnalysis, without its
    optional per-lane conflict filter. Returned barriers are boundary barriers;
    internal scratch barriers are represented separately by the lowering plan.
    """
    pending = []
    result = []
    names = set()

    def checked_ranges(values):
        ranges = []
        for pair in values:
            if (
                len(pair) != 2
                or any(isinstance(v, bool) or not isinstance(v, int) for v in pair)
                or not 0 <= pair[0] < pair[1]
            ):
                raise ValueError("Require nonempty half-open shared byte intervals")
            ranges.append(tuple(pair))
        return ranges

    for op in operations:
        name, kind = op["id"], op["kind"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Require unique operation IDs")
        names.add(name)
        if kind == "cta_barrier":
            if op.get("reads") or op.get("writes") or "scratch" in op:
                raise ValueError("Barrier cannot have memory effects")
            pending.clear()
            result.append(dict(id=name, inserted=False, explicit=True, hazards=[]))
            continue
        if kind not in {"access", "scratch"}:
            raise ValueError("Unsupported shared operation or control flow")
        reads, writes = (
            checked_ranges(op.get("reads", [])),
            checked_ranges(op.get("writes", [])),
        )
        if kind == "scratch":
            if reads or writes:
                raise ValueError(
                    "Scratch effects must be specified only by its interval"
                )
            writes = checked_ranges([op["scratch"]])
            if op.get("internal_sync") not in {"cta", "warp"}:
                raise ValueError("Require explicit scratch synchronization scope")
        elif "scratch" in op or "internal_sync" in op:
            raise ValueError("Unexpected scratch attributes on memory access")
        current = [("read", region) for region in reads] + [
            ("write", region) for region in writes
        ]
        hazards = []
        for old_name, old_mode, old_range in pending:
            for mode, region in current:
                if old_mode == mode == "read":
                    continue
                overlap = [max(old_range[0], region[0]), min(old_range[1], region[1])]
                if overlap[0] < overlap[1]:
                    hazards.append(
                        dict(
                            producer=old_name,
                            kind={
                                ("write", "read"): "RAW",
                                ("read", "write"): "WAR",
                                ("write", "write"): "WAW",
                            }[(old_mode, mode)],
                            overlap=overlap,
                        )
                    )
        inserted = bool(hazards)
        if inserted or kind == "scratch" and op["internal_sync"] == "cta":
            pending.clear()
        if kind == "scratch":
            current.append(("read", writes[0]))
        pending.extend((name, mode, region) for mode, region in current)
        result.append(dict(id=name, inserted=inserted, explicit=False, hazards=hazards))
    return dict(
        boundary_barrier_count=sum(r["inserted"] for r in result), operations=result
    )
