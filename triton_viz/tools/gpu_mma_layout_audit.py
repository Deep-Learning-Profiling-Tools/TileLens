"""Control-only validation of source-derived single and chained MMA-v2 layouts."""

import argparse
import json
import re
from pathlib import Path

from triton_viz.performance.gpu_layout import (
    mma_v2_warp_layout,
    mma_v2_row_reduction,
    mma_v2_issued_work,
)
from triton_viz.tools.gpu_control_resource_audit import audit as resource_audit
from triton_viz.tools.gpu_cost_model_pipeline import _write


def reduction_instruction_labels(ptx):
    """Control labels from pinned standard.py max/sum locations, never features.

    Keep all other shuffles separate; casting/layout conversions can emit many
    shuffles that must not be silently charged as reductions.
    """
    files = {
        int(index): Path(path).name
        for index, path in re.findall(r'\.file\s+(\d+)\s+"([^"]+)"', ptx)
    }
    counts = {
        op: dict(shuffles=0, barriers=0, leading_barriers=0)
        for op in ("max", "sum", "other")
    }
    shared_store_seen = set()
    locations = {191: "max", 293: "sum"}  # Triton 3.7 standard.py
    active = "other"
    seen = set()
    for raw in ptx.splitlines():
        line = raw.split("//", 1)[0]
        loc = re.search(r"\.loc\s+(\d+)\s+(\d+)\s", line)
        if loc:
            file_id, lineno = map(int, loc.groups())
            active = (
                locations.get(lineno, "other")
                if files.get(file_id) == "standard.py"
                else "other"
            )
            seen.add(active)
        if re.search(r"\bshfl\.sync\.", line):
            counts[active]["shuffles"] += 1
        if re.search(r"\bst\.shared\.", line):
            shared_store_seen.add(active)
        if re.search(r"\bbar\.sync\s", line):
            counts[active]["barriers"] += 1
            if active not in shared_store_seen:
                counts[active]["leading_barriers"] += 1
    if not {"max", "sum"} <= seen:
        raise ValueError("Missing pinned control reduction locations")
    return counts


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


def source_instruction_plan(source, *, compiler_version):
    """Conditional instruction slots for the same declared static-region scope."""
    layouts = source_plans(source, compiler_version=compiler_version)
    if layouts is None:
        return None
    groups = {}
    for dot in source["dot_ancestry"]:
        groups.setdefault(tuple(dot["program"]), []).append(dot)
    signatures = {
        tuple(tuple(tuple(s) for s in dot["input_shapes"][:2]) for dot in dots)
        for dots in groups.values()
    }
    if len(signatures) != 1:
        raise ValueError("Nonuniform source dot shapes")
    precision = source["source_precision"][0]
    dtype = "tf32" if precision[0] == "fp32" else precision[0]
    result = []
    for (a, b), layout in zip(next(iter(signatures)), layouts):
        if a[1] != b[0]:
            raise ValueError("Invalid source dot contraction")
        work = mma_v2_issued_work(
            a[0],
            b[1],
            a[1],
            int(source["source_features"]["threads_per_program"] // 32),
            input_dtype=dtype,
            chained_dot=len(layouts) == 2,
            compiler_version=compiler_version,
        )
        if work["warp_layout"] != layout:
            raise ValueError("Source layout policy drift")
        result.append(work)
    return result


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
        work = source_instruction_plan(
            source, compiler_version=manifest["packages"]["triton"]
        )
        cfg = resource["sass_backedges"]
        straight = cfg is not None and cfg["supported"] and not cfg["regions"]
        predicted_instructions = sum(p["instructions_per_warp"] for p in work)
        emitted_instructions = resource["static_sass_counts"].get("HMMA", 0)
        rows.append(
            dict(
                case=case,
                applicable=True,
                prediction=plan,
                observed=observed,
                exact_match=plan == observed,
                instruction_plan=work,
                instruction_count_applicable=straight,
                predicted_hmma_instructions_per_warp=predicted_instructions,
                emitted_static_hmma_instructions=emitted_instructions,
                exact_instruction_count_match=straight
                and predicted_instructions == emitted_instructions,
            )
        )
        if case["kind"] == "structure_composition" and case["variant"] >= 1:
            first = source["dot_ancestry"][0]
            m, n = first["input_shapes"][0][0], first["input_shapes"][1][1]
            reduction = mma_v2_row_reduction(
                m,
                n,
                int(source["source_features"]["threads_per_program"] // 32),
                chained_dot=len(plan) == 2,
                compiler_version=manifest["packages"]["triton"],
            )
            labels = reduction_instruction_labels(compiled["artifacts"]["ptx"])
            rows[-1].update(
                reduction_prediction=reduction,
                reduction_labels=labels,
                exact_reduction_shuffle_match=all(
                    labels[op]["shuffles"] == reduction["total_shuffles"]
                    for op in ("max", "sum")
                ),
                exact_reduction_match=all(
                    {k: labels[op][k] for k in ("shuffles", "barriers")}
                    == dict(
                        shuffles=reduction["total_shuffles"],
                        barriers=reduction["reduction_barriers"],
                    )
                    for op in ("max", "sum")
                ),
            )
    return dict(
        role="control",
        eligible_for_fit=False,
        count=len(rows),
        compared_count=sum(r["applicable"] for r in rows),
        exact_count=sum(r.get("exact_match", False) for r in rows),
        instruction_compared_count=sum(
            r.get("instruction_count_applicable", False) for r in rows
        ),
        instruction_exact_count=sum(
            r.get("exact_instruction_count_match", False) for r in rows
        ),
        warp_replication_count=sum(
            any(
                p["warp_layout_exceeds_geometry"] for p in r.get("instruction_plan", [])
            )
            for r in rows
        ),
        reduction_compared_count=sum("exact_reduction_match" in r for r in rows),
        reduction_exact_count=sum(r.get("exact_reduction_match", False) for r in rows),
        reduction_shuffle_exact_count=sum(
            r.get("exact_reduction_shuffle_match", False) for r in rows
        ),
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
