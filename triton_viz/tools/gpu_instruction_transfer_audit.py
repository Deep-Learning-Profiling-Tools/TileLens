"""Control-only transfer of instruction work, not compiler-at-prediction pricing.

Measured warp instruction totals are normalized by source program/warp counts.
The source query contains no measured instruction, cache, resource or time label.
Single/repeated dot support remains explicit; generic composed regions need
their own lowering mapping rather than invented zero instruction overhead.
"""

import math

from triton_viz.tools.gpu_spill_transfer_audit import (
    FEATURES,
    dot_instructions,
    request_candidates,
)

LABELS = ("LDL", "STL", "executed", "issued")
WORK_LABELS = ("LDL", "STL", "other")


def decompose_work(row):
    """Disjoint training-label components; never clamp inconsistent evidence."""
    if row.get("role") != "control":
        raise ValueError("Instruction decomposition requires control labels")
    measured = row["instructions_per_warp"]
    if any(not math.isfinite(measured[k]) or measured[k] < 0 for k in LABELS):
        raise ValueError("Invalid instruction label")
    other = measured["executed"] - measured["LDL"] - measured["STL"] - scale(row)
    if other < 0:
        raise ValueError("Instruction decomposition has negative residual work")
    return dict(LDL=measured["LDL"], STL=measured["STL"], other=other)


def scale(row):
    if "source_execution" in row:
        return execution_scale(row)
    value = dot_instructions(row) * row["source_features"]["dots_per_program"]
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Invalid source dot instruction scale")
    return value


def execution_scale(row):
    """Experimental observed-region work, without reading compiler/query labels.

    Requires uniform program work. Compiler-region eligibility is still a
    separate admission check; this function does not certify arbitrary kernels.
    """
    from triton_viz.performance.gpu_source_regions import (
        conditional_mma_work,
        dot_execution_regions,
    )

    if row["compiler_version"] != "3.7.0":
        raise ValueError("Unverified compiler expansion policy")
    source = row["source_execution"]
    regions = dot_execution_regions(source["dot_ancestry"], source["loop_trace"])
    count = row["source_program_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("Invalid source program count")
    threads = row["source_features"]["threads_per_program"]
    if (
        isinstance(threads, bool)
        or not isinstance(threads, int)
        or threads <= 0
        or threads % 32
    ):
        raise ValueError("Invalid source thread count")
    if row["source_precision"] == [["fp32", "fp32", "ieee"]]:
        programs = {}
        for region in regions["regions"]:
            program = tuple(region["program"])
            for a, b in region["dot_shapes"]:
                if (
                    len(a) != 2
                    or len(b) != 2
                    or a[1] != b[0]
                    or any(
                        isinstance(v, bool) or not isinstance(v, int) or v <= 0
                        for v in (*a, *b)
                    )
                ):
                    raise ValueError("Invalid source dot geometry")
                work = a[0] * a[1] * b[1]
                if work % threads:
                    raise ValueError("Partial scalar expansion")
                programs[program] = programs.get(program, 0) + work // threads
    else:
        plan = conditional_mma_work(
            regions,
            precision=row["source_precision"],
            warps=threads // 32,
            compiler_version=row["compiler_version"],
        )
        programs = {
            tuple(p["program"]): p["instructions_per_warp"] for p in plan["programs"]
        }
    if not programs or len(programs) != count or len(set(programs.values())) != 1:
        raise ValueError("Require complete uniform source program work")
    if regions["dot_count"] / count != row["source_features"]["dots_per_program"]:
        raise ValueError("Source dot observation count mismatch")
    value = next(iter(programs.values()))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Invalid source dot instruction scale")
    return value


def _predict(training, source, labels):
    if not training or any(r.get("role") != "control" for r in training):
        raise ValueError("Require control instruction training")
    for row in [*training, source]:
        if any(
            not math.isfinite(row["source_features"][k])
            or row["source_features"][k] < 0
            for k in FEATURES
        ):
            raise ValueError("Invalid source descriptor")
    for row in training:
        if any(
            not math.isfinite(row["instructions_per_warp"][k])
            or row["instructions_per_warp"][k] < 0
            for k in labels
        ):
            raise ValueError("Invalid instruction label")
    domain = {
        k: (
            min(r["source_features"][k] for r in training),
            max(r["source_features"][k] for r in training),
        )
        for k in FEATURES
    }
    candidates, reasons = request_candidates(training, source, separate_single_dot=True)
    distances = [
        sum(
            abs(source["source_features"][k] - r["source_features"][k]) / (hi - lo)
            if hi > lo
            else float(source["source_features"][k] != lo)
            for k, (lo, hi) in domain.items()
        )
        for r in candidates
    ]
    nearest = [r for r, d in zip(candidates, distances) if d == min(distances)]
    predicted = {
        k: scale(source)
        * sum(r["instructions_per_warp"][k] / scale(r) for r in nearest)
        / len(nearest)
        for k in labels
    }
    return dict(
        instructions_per_warp=predicted,
        neighbors=[r["case"]["id"] for r in nearest],
        ood_reasons=reasons
        + (
            ["instruction_region_lowering_conditional"]
            if "source_execution" in source
            else []
        )
        + [
            f"instruction_domain:{k}"
            for k, (lo, hi) in domain.items()
            if not lo <= source["source_features"][k] <= hi
        ],
    )


def predict(training, source):
    return _predict(training, source, LABELS)


def predict_work(training, source):
    # Transform only training labels. The query never needs its own compiled
    # instruction count and cannot leak it through the decomposition.
    decomposed = [
        {**row, "instructions_per_warp": decompose_work(row)} for row in training
    ]
    return _predict(decomposed, source, WORK_LABELS)


def validate(rows, *, components=False):
    if not rows or any(r.get("role") != "control" for r in rows):
        raise ValueError("Require control instruction rows")
    if len({r["case"]["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate controls")
    owners = {}
    for row in rows:
        for a, b in row["dot_shapes"]:
            key = (tuple(a), tuple(b))
            group = row["case"]["cv_group"]
            if key in owners and owners[key] != group:
                raise ValueError("Shared geometry crosses validation partitions")
            owners[key] = group
        if (
            isinstance(row["source_program_count"], bool)
            or not isinstance(row["source_program_count"], int)
            or row["source_program_count"] < 1
        ):
            raise ValueError("Invalid source program count")
    groups = sorted({r["case"]["cv_group"] for r in rows})
    if len(groups) < 3:
        raise ValueError("Need three geometry groups")
    predictions = []
    for group in groups:
        training = [r for r in rows if r["case"]["cv_group"] != group]
        for row in rows:
            if row["case"]["cv_group"] != group:
                continue
            predictions.append(
                dict(
                    case=row["case"],
                    actual=decompose_work(row)
                    if components
                    else row["instructions_per_warp"],
                    warp_count=row["source_program_count"]
                    * row["source_features"]["threads_per_program"]
                    / 32,
                    training_ids=[r["case"]["id"] for r in training],
                    **(predict_work if components else predict)(training, row),
                )
            )
    metrics = {}
    for name in WORK_LABELS if components else LABELS:
        measured = sum(r["actual"][name] * r["warp_count"] for r in predictions)
        difference = sum(
            abs(r["actual"][name] - r["instructions_per_warp"][name]) * r["warp_count"]
            for r in predictions
        )
        metrics[name] = dict(
            instruction_count_wape_pct=100 * difference / measured
            if measured
            else None,
            mae_instructions_per_warp=sum(
                abs(r["actual"][name] - r["instructions_per_warp"][name])
                for r in predictions
            )
            / len(predictions),
            false_positive=sum(
                r["actual"][name] == 0 and r["instructions_per_warp"][name] > 0
                for r in predictions
            ),
            false_negative=sum(
                r["actual"][name] > 0 and r["instructions_per_warp"][name] == 0
                for r in predictions
            ),
        )
    return dict(
        role="control",
        count=len(rows),
        components=components,
        eligible_for_fit=False,
        released_model=None,
        rows=predictions,
        metrics=metrics,
        ood_count=sum(bool(r["ood_reasons"]) for r in predictions),
        caveat="Instruction-work transfer only, not cache or latency validation. All controls retained; no target read.",
    )
