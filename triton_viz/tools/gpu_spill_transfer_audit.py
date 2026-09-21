"""Control-only source-to-local-request transfer, with zero-traffic rows retained.

Direct counter labels bypass allocation bytes. Source-normalized nearest-control
diagnostics are not cache-miss models, latency fits or a released predictor.
"""

import math

from triton_viz.tools.gpu_resource_transfer_audit import FEATURE_SETS

FEATURES = FEATURE_SETS["initial_layout"]
OPS = ("LDL", "STL")


def dot_instructions(row):
    """Conditional pinned pure-dot expansion per thread, per source dot.

    IEEE uses scalar FMA; other supported precisions use the control-validated
    MMA-v2 instruction shapes. This is not generic MMA eligibility inference.
    """
    if row["compiler_version"] != "3.7.0":
        raise ValueError("Unverified compiler expansion policy")
    shapes = row["dot_shapes"]
    if len(shapes) != 1:
        raise ValueError("Require one homogeneous dot geometry")
    (m, k), (kb, n) = shapes[0]
    threads = row["source_features"]["threads_per_program"]
    if (
        any(
            isinstance(v, bool) or not isinstance(v, int) or v <= 0
            for v in (m, k, kb, n, threads)
        )
        or k != kb
        or threads % 32
    ):
        raise ValueError("Invalid source geometry")
    precision = row["source_precision"]
    if precision == [["fp32", "fp32", "ieee"]]:
        if m * n * k % threads:
            raise ValueError("Partial scalar expansion")
        return m * n * k / threads
    ik = (
        8
        if precision == [["fp32", "fp32", "tf32"]]
        else 16
        if precision in ([["bf16", "bf16", "ieee"]], [["fp16", "fp16", "ieee"]])
        else None
    )
    if (
        ik is None
        or m % 16
        or n % 8
        or k % ik
        or (m * n * k) % (16 * 8 * ik * (threads // 32))
    ):
        raise ValueError("Unsupported or partial MMA expansion")
    return m * n * k / (16 * 8 * ik * (threads // 32))


def validate(rows, *, normalization):
    if normalization not in {"dot", "instruction"}:
        raise ValueError("Unknown normalization")
    if not rows or any(r.get("role") != "control" for r in rows):
        raise ValueError("Require control rows only")
    if len({r["case"]["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate control")
    for row in rows:
        if any(
            not math.isfinite(row["source_features"][k])
            or row["source_features"][k] < 0
            for k in FEATURES
        ) or any(
            not math.isfinite(row["counter_bytes_per_thread"][op])
            or row["counter_bytes_per_thread"][op] < 0
            for op in OPS
        ):
            raise ValueError("Invalid source/counter observation")
        if (
            row["source_features"]["dots_per_program"] <= 0
            or row["source_program_count"] <= 0
            or row.get("ood_reasons")
        ):
            raise ValueError("Unsupported source control; no removal")
    groups = sorted({r["case"]["cv_group"] for r in rows})
    if len(groups) < 3:
        raise ValueError("Require three geometry groups")

    def scale(row):
        return row["source_features"]["dots_per_program"] * (
            dot_instructions(row) if normalization == "instruction" else 1
        )

    results = []
    for held in groups:
        training = [r for r in rows if r["case"]["cv_group"] != held]
        domain = {
            k: (
                min(r["source_features"][k] for r in training),
                max(r["source_features"][k] for r in training),
            )
            for k in FEATURES
        }
        for row in rows:
            if row["case"]["cv_group"] != held:
                continue
            candidates = [
                r for r in training if r["source_precision"] == row["source_precision"]
            ]
            if not candidates:
                raise ValueError("Unseen precision; all controls retained")
            distances = [
                sum(
                    abs(row["source_features"][k] - r["source_features"][k]) / (hi - lo)
                    if hi > lo
                    else float(row["source_features"][k] != lo)
                    for k, (lo, hi) in domain.items()
                )
                for r in candidates
            ]
            nearest = [r for r, d in zip(candidates, distances) if d == min(distances)]
            predicted = {
                op: scale(row)
                * sum(r["counter_bytes_per_thread"][op] / scale(r) for r in nearest)
                / len(nearest)
                for op in OPS
            }
            results.append(
                dict(
                    case=row["case"],
                    predicted=predicted,
                    actual=row["counter_bytes_per_thread"],
                    payload_scale=row["source_program_count"]
                    * row["source_features"]["threads_per_program"]
                    / 32,
                    training_ids=[r["case"]["id"] for r in training],
                    neighbors=[r["case"]["id"] for r in nearest],
                    ood_reasons=[
                        f"outside_training_domain:{k}"
                        for k, (lo, hi) in domain.items()
                        if not lo <= row["source_features"][k] <= hi
                    ],
                )
            )
    metrics = {}
    for op in OPS:
        absolute = sum(
            abs(r["predicted"][op] - r["actual"][op]) * r["payload_scale"]
            for r in results
        )
        total = sum(r["actual"][op] * r["payload_scale"] for r in results)
        metrics[op] = dict(
            payload_sector_wape_pct=100 * absolute / total if total else None,
            mae_bytes_per_thread=sum(
                abs(r["predicted"][op] - r["actual"][op]) for r in results
            )
            / len(results),
            false_positive=sum(
                r["actual"][op] == 0 and r["predicted"][op] > 0 for r in results
            ),
            false_negative=sum(
                r["actual"][op] > 0 and r["predicted"][op] == 0 for r in results
            ),
        )
    return dict(
        role="control",
        eligible_for_fit=False,
        released_model=None,
        count=len(results),
        normalization=normalization,
        metrics=metrics,
        rows=results,
        ood_count=sum(bool(r["ood_reasons"]) for r in results),
        caveat="Local request payload, not cache misses or latency. Conditional pure-dot instruction expansion. All zero/nonzero controls retained; no target read.",
    )
