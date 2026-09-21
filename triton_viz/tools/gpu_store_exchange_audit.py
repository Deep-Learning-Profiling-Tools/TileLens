"""Control-only audit of source-derived IEEE contiguous-store exchange rounds."""

import argparse
import json
import re
from pathlib import Path

from triton_viz.performance.gpu_layout import ieee_row_store_exchange
from triton_viz.tools.gpu_control_resource_audit import audit as resource_audit
from triton_viz.tools.gpu_cost_model_pipeline import _write


def epilogue_exchange(ptx):
    """Count static shared-store bursts after the last dot FMA, not barriers.

    Restricted to pure-dot controls: later async input prefetch is distinct from
    explicit st.shared output exchange. Not arbitrary kernel phase inference.
    """
    lines = [line.split("//", 1)[0] for line in ptx.splitlines()]
    fmas = [
        i for i, line in enumerate(lines) if re.search(r"\bfma\.rn\.f32(?:x2)?\s", line)
    ]
    if not fmas:
        raise ValueError("Missing IEEE dot body")
    rounds, stores, loads, burst = 0, 0, 0, False
    for line in lines[fmas[-1] + 1 :]:
        if re.search(r"\bbar\.sync\s", line):
            rounds += int(burst)
            burst = False
        if re.search(r"\bst\.shared\.", line):
            burst = True
            stores += 1
        if re.search(r"\bld\.shared\.", line):
            loads += 1
    return dict(
        shared_store_rounds=rounds + int(burst),
        shared_store_instructions=stores,
        shared_load_instructions=loads,
    )


def audit(resource_root, source_root):
    resources = resource_audit(resource_root)
    manifest = json.loads((resource_root / "manifest.json").read_text())
    sources = json.loads((source_root / "manifest.json").read_text())
    if (
        not resources["complete"]
        or sources.get("role") != "control"
        or sources["cases"] != manifest["cases"]
    ):
        raise ValueError("Require complete matching control manifests")
    rows = []
    for resource in resources["rows"]:
        case = resource["case"]
        source = json.loads(
            (source_root / "controls" / (case["id"] + ".json")).read_text()
        )
        if (
            source.get("role") != "control"
            or source["case"] != case
            or source.get("compile_and_cuda_forbidden") is not True
            or source.get("numerical_validation") != "passed"
        ):
            raise ValueError("Unverified source observation")
        if case["kind"] != "geometry_dot" or source["source_precision"] != [
            ["fp32", "fp32", "ieee"]
        ]:
            rows.append(
                dict(
                    case=case,
                    applicable=False,
                    reason="Only pure IEEE dot with declared fresh contiguous FP32 output",
                )
            )
            continue
        if len(source["dot_shapes"]) != 1 or source["ood_reasons"]:
            raise ValueError("Unsupported source geometry")
        a, b = source["dot_shapes"][0]
        threads = source["source_features"]["threads_per_program"]
        if a[1] != b[0] or threads % 32:
            raise ValueError("Invalid source geometry or warp count")
        # This output allocation contract is specific to prepare(geometry_dot),
        # not inferred for arbitrary kernels or from emitted compiler layouts.
        plan = ieee_row_store_exchange(
            a[0],
            b[1],
            int(threads // 32),
            compiler_version=manifest["packages"]["triton"],
            alignment_bytes=16,
        )
        compiled = json.loads(
            (resource_root / "controls" / (case["id"] + ".json")).read_text()
        )
        observed = epilogue_exchange(compiled["artifacts"]["ptx"])
        rows.append(
            dict(
                case=case,
                applicable=True,
                prediction=plan,
                observed=observed,
                exact_round_match=None
                if plan["shared_rounds"] is None
                else plan["shared_rounds"] == observed["shared_store_rounds"],
            )
        )
    compared = [r for r in rows if r.get("exact_round_match") is not None]
    return dict(
        role="control",
        count=len(rows),
        rows=rows,
        compared_count=len(compared),
        exact_count=sum(r["exact_round_match"] for r in compared),
        eligible_for_fit=False,
        caveat="Control source-derived output-exchange rule only. Core conversion barriers exclude surrounding hazards and pipeline drains; neither is total-kernel latency or spill allocation.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Use a fresh exchange audit output")
    result = audit(args.resource_root, args.source_root)
    _write(args.output, result)
    print({k: result[k] for k in ("count", "compared_count", "exact_count")})


if __name__ == "__main__":
    main()
