"""Check source dot instruction expansion against declared control artifacts.

Control-only diagnostic: PTX instruction shapes provide expansion-table entries;
SASS supplies independent emitted instruction counts. No latency/target input.
"""

import argparse
import json
import re
from pathlib import Path

from triton_viz.tools.gpu_control_resource_audit import audit as resource_audit
from triton_viz.tools.gpu_cost_model_pipeline import _write


def audit(resource_root, source_root):
    resources = resource_audit(resource_root)
    manifest = json.loads((source_root / "manifest.json").read_text())
    if manifest.get("role") != "control" or not resources["complete"]:
        raise ValueError("Require complete declared control artifacts")
    cases = {c["id"]: c for c in manifest["cases"]}
    if len(cases) != len(manifest["cases"]) or set(cases) != {
        r["case"]["id"] for r in resources["rows"]
    }:
        raise ValueError("Control manifests differ")
    rows = []
    for resource in resources["rows"]:
        case = resource["case"]
        source = json.loads(
            (source_root / "controls" / (case["id"] + ".json")).read_text()
        )
        if (
            source.get("role") != "control"
            or source["case"] != case
            or cases[case["id"]] != case
        ):
            raise ValueError("Source control identity mismatch")
        if (
            source.get("numerical_validation") != "passed"
            or source.get("compile_and_cuda_forbidden") is not True
        ):
            raise ValueError("Unverified precompile source observation")
        if case["kind"] != "geometry_dot":
            rows.append(
                dict(
                    case=case,
                    applicable=False,
                    reason="Pure-dot expansion check; composition retained for separate phase expansion",
                )
            )
            continue
        compiled = json.loads(
            (resource_root / "controls" / (case["id"] + ".json")).read_text()
        )
        shapes = source["dot_shapes"]
        if len(shapes) != 1 or source["ood_reasons"]:
            raise ValueError("Pure-dot control requires one supported source dot shape")
        a, b = shapes[0]
        fmas = a[0] * a[1] * b[1]
        operations = source["operation_counts"]["dot"] / source["program_count"]
        loops = resource["static_ttgir_loop_count"]
        if loops not in (0, 1):
            raise ValueError("Nested compiler loops require a separate expansion model")
        body_count = 1 if loops else operations
        instructions = sorted(
            set(
                re.findall(
                    r"mma\.sync\.aligned\.m(\d+)n(\d+)k(\d+)\.row\.col\.f32\.([a-z0-9]+)\.([a-z0-9]+)\.f32",
                    compiled["artifacts"]["ptx"],
                )
            )
        )
        if instructions:
            if len(instructions) != 1:
                raise ValueError(
                    "Mixed MMA instruction forms require separate expansion"
                )
            m, n, k, dtype_a, dtype_b = instructions[0]
            volume = int(m) * int(n) * int(k)
            if a[0] % int(m) or b[1] % int(n) or a[1] % int(k):
                raise ValueError(
                    "Partial instruction tiles require explicit padded expansion"
                )
            expected = (
                body_count
                * fmas
                / volume
                / (source["source_features"]["threads_per_program"] / 32)
            )
            opcode = "HMMA"
            rule = dict(
                kind="tensor",
                instruction_shape=[int(m), int(n), int(k)],
                input_dtypes=[dtype_a, dtype_b],
            )
        else:
            if source["source_precision"] != [["fp32", "fp32", "ieee"]]:
                raise ValueError("Unknown non-MMA dot lowering")
            expected = (
                body_count * fmas / source["source_features"]["threads_per_program"]
            )
            opcode, rule = "FFMA", dict(kind="simt_fp32")
        actual = resource["static_sass_counts"].get(opcode, 0)
        ptx_counts = resource["static_ptx_counts"]
        # f32x2 expresses two scalar FMAs per PTX instruction. This is a typed
        # lane expansion, not an empirical correction fitted to SASS counts.
        ptx_actual = (
            ptx_counts["mma"]
            if instructions
            else ptx_counts["fp32_fma"] + 2 * ptx_counts["fp32_fma_x2"]
        )
        rows.append(
            dict(
                case=case,
                applicable=True,
                rule=rule,
                opcode=opcode,
                retained_compiler_loops=loops,
                source_dots_per_program=operations,
                predicted_static_instructions=expected,
                emitted_static_instructions=actual,
                exact_match=expected == actual,
                emitted_static_ptx_instructions=ptx_actual,
                ptx_exact_match=expected == ptx_actual,
                sass_backedges=resource["sass_backedges"],
                local_bytes_per_thread=resource["local_bytes_per_thread"],
            )
        )
    applicable = [r for r in rows if r["applicable"]]
    return dict(
        role="control",
        count=len(rows),
        rows=rows,
        checked_count=len(applicable),
        exact_count=sum(r["exact_match"] for r in applicable),
        ptx_exact_count=sum(r["ptx_exact_match"] for r in applicable),
        eligible_for_fit=False,
        caveat="Control instruction expansion check, not a latency fit or a source-only prediction of loop retention/spill instructions.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Use a new audit output")
    report = audit(args.resource_root, args.source_root)
    _write(args.output, report)
    print(
        {
            k: report[k]
            for k in ("count", "checked_count", "ptx_exact_count", "exact_count")
        }
    )


if __name__ == "__main__":
    main()
