"""Control-only validation of source-derived single and chained MMA-v2 layouts."""

import argparse
import json
import re
from pathlib import Path

from triton_viz.performance.gpu_layout import mma_v2_warp_layout
from triton_viz.tools.gpu_control_resource_audit import audit as resource_audit
from triton_viz.tools.gpu_cost_model_pipeline import _write


def compiled_dot_warps(ttgir):
    layouts = {
        alias: [int(m), int(n)]
        for alias, m, n in re.findall(
            r"^(#[\w]+) = #ttg\.nvidia_mma<\{versionMajor = 2, versionMinor = 0, "
            r"warpsPerCTA = \[(\d+), (\d+)\]",
            ttgir,
            re.M,
        )
    }
    result = []
    for line in ttgir.splitlines():
        if re.search(r"\btt\.dot\s", line):
            output = re.search(r"-> tensor<\d+x\d+xf32, (#[\w]+)>", line)
            if not output or output[1] not in layouts:
                raise ValueError("Expected rank-2 MMA-v2 control dot")
            result.append(layouts[output[1]])
    if not result:
        raise ValueError("Missing control dot")
    return result


def source_plans(source, *, compiler_version):
    """Control scope supplies static-region proof, never compiler labels.

    Only repeat-1 geometry and the declared straight-line composition kernel.
    Dynamic ancestry alone is insufficient for arbitrary loop/region inference.
    """
    case = source["case"]
    precision = source["source_precision"]
    if len(precision) != 1 or precision[0] not in (
        ["fp32", "fp32", "tf32"],
        ["fp16", "fp16", "ieee"],
        ["bf16", "bf16", "ieee"],
    ):
        return None
    if (
        case["kind"] not in {"geometry_dot", "structure_composition"}
        or case["repeat"] != 1
    ):
        return None
    if source["ood_reasons"]:
        raise ValueError("Unsupported source observation")
    groups = {}
    for dot in source["dot_ancestry"]:
        groups.setdefault(tuple(dot["program"]), []).append(dot)
    if len(groups) != source["program_count"]:
        raise ValueError("Missing program ancestry")
    expected = (
        2 if case["kind"] == "structure_composition" and case["variant"] == 2 else 1
    )
    plans = []
    for dots in groups.values():
        if len(dots) != expected or dots[0]["ancestor_dot_seqs"]:
            raise ValueError("Static control chain does not match observed ancestry")
        chained = expected == 2
        if chained and dots[1]["ancestor_dot_seqs"] != [dots[0]["seq"]]:
            raise ValueError("Missing source producer-consumer edge")
        shapes = [(d["input_shapes"][0][0], d["input_shapes"][1][1]) for d in dots]
        if len(set(shapes)) != 1:
            raise ValueError("Heterogeneous chain requires layout propagation")
        threads = source["source_features"]["threads_per_program"]
        if threads % 32:
            raise ValueError("Nonintegral warp count")
        plan = [
            mma_v2_warp_layout(
                m,
                n,
                int(threads // 32),
                chained_dot=chained,
                compiler_version=compiler_version,
            )
            for m, n in shapes
        ]
        plans.append(plan)
    if not plans or any(p != plans[0] for p in plans):
        raise ValueError("Nonuniform source program layouts")
    return plans[0]


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
            raise ValueError("Unverified source control")
        plan = source_plans(source, compiler_version=manifest["packages"]["triton"])
        if plan is None:
            rows.append(
                dict(
                    case=case,
                    applicable=False,
                    reason="Outside straight-line homogeneous MMA-v2 scope",
                )
            )
            continue
        compiled = json.loads(
            (resource_root / "controls" / (case["id"] + ".json")).read_text()
        )
        observed = compiled_dot_warps(compiled["artifacts"]["ttgir"])
        rows.append(
            dict(
                case=case,
                applicable=True,
                prediction=plan,
                observed=observed,
                exact_match=plan == observed,
            )
        )
    return dict(
        role="control",
        eligible_for_fit=False,
        count=len(rows),
        compared_count=sum(r["applicable"] for r in rows),
        exact_count=sum(r.get("exact_match", False) for r in rows),
        rows=rows,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Use fresh audit output")
    result = audit(args.resource_root, args.source_root)
    _write(args.output, result)
    print({k: result[k] for k in ("count", "compared_count", "exact_count")})


if __name__ == "__main__":
    main()
