"""Validate a source-only initial layout rule against control compiler labels."""

import argparse
import json
from pathlib import Path

from triton_viz.performance.gpu_layout import initial_ieee_dot_layout
from triton_viz.tools.gpu_control_resource_audit import audit as resource_audit
from triton_viz.tools.gpu_cost_model_pipeline import _write


def audit(resource_root, source_root):
    resources = resource_audit(resource_root)
    source_manifest = json.loads((source_root / "manifest.json").read_text())
    compiled_manifest = json.loads((resource_root / "manifest.json").read_text())
    if (
        source_manifest.get("role") != "control"
        or not resources["complete"]
        or source_manifest["cases"] != compiled_manifest["cases"]
    ):
        raise ValueError("Require identical complete declared controls")
    version = compiled_manifest["packages"]["triton"]
    rows = []
    for compiled in resources["rows"]:
        case = compiled["case"]
        source = json.loads(
            (source_root / "controls" / (case["id"] + ".json")).read_text()
        )
        if (
            source.get("role") != "control"
            or source["case"] != case
            or source.get("numerical_validation") != "passed"
            or source.get("compile_and_cuda_forbidden") is not True
        ):
            raise ValueError("Unverified source control")
        row = dict(case=case)
        if source["source_precision"] != [["fp32", "fp32", "ieee"]]:
            rows.append(
                dict(
                    **row,
                    applicable=False,
                    reason="Initial SIMT IEEE rule; tensor layout mapping remains separate",
                )
            )
            continue
        threads = source["source_features"]["threads_per_program"]
        if threads % 32:
            raise ValueError("Nonintegral source warp count")
        predictions = {}
        for a, b in source["dot_shapes"]:
            m, k = a
            kb, n = b
            if kb != k:
                raise ValueError("Invalid source dot shape")
            predictions[m, n, k] = initial_ieee_dot_layout(
                m, n, k, int(threads // 32), compiler_version=version
            )
        dots = []
        for emitted in compiled["blocked_dot_fragments"]:
            if not emitted["supported"]:
                dots.append(dict(supported=False, reason=emitted["reason"]))
                continue
            key = tuple(emitted["shape"])
            if key not in predictions:
                raise ValueError("Compiler dot has no corresponding source geometry")
            predicted = predictions[key]
            exact = all(
                predicted[p] == emitted["layout"].get(e)
                for p, e in (
                    ("size_per_thread", "sizePerThread"),
                    ("threads_per_warp", "threadsPerWarp"),
                    ("warps_per_cta", "warpsPerCTA"),
                    ("order", "order"),
                )
            )
            dots.append(
                dict(
                    supported=True,
                    source_prediction=predicted,
                    emitted=emitted,
                    exact_layout_match=exact,
                )
            )
        rows.append(dict(**row, applicable=True, dots=dots))
    dots = [d for r in rows if r["applicable"] for d in r["dots"]]
    return dict(
        role="control",
        count=len(rows),
        rows=rows,
        dot_count=len(dots),
        exact_layout_count=sum(d.get("exact_layout_match", False) for d in dots),
        eligible_for_fit=False,
        caveat="Versioned source-only initial SIMT layout check. Fully materialized fragments are not physical peak registers or spill predictions. All tensor controls retained as separately unsupported.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Use a fresh layout audit output")
    result = audit(args.resource_root, args.source_root)
    _write(args.output, result)
    print({k: result[k] for k in ("count", "dot_count", "exact_layout_count")})


if __name__ == "__main__":
    main()
