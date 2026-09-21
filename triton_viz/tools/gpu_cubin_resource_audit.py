"""Reproduce control cubins offline and compare allocation declarations to driver.

No CUDA context, kernel launch, latency label or target artifact is used. A
matching cubin declaration is not a driver measurement for an impossible launch.
"""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from triton_viz.tools.gpu_cost_model_pipeline import _write


def allocation_declaration(text, *, kernel_name):
    functions = re.findall(r"^\s*Function ([^:\n]+):\s*\n([^\n]+)", text, re.M)
    if len(functions) != 1 or functions[0][0] != kernel_name:
        raise ValueError("Require exactly the declared control kernel")
    fields = re.findall(r"\b(REG|STACK|LOCAL):(\d+)\b", functions[0][1])
    if len(fields) != 3 or {k for k, _ in fields} != {"REG", "STACK", "LOCAL"}:
        raise ValueError("Missing or ambiguous cubin allocation declaration")
    values = {k: int(v) for k, v in fields}
    return dict(
        registers_per_thread=values["REG"],
        local_bytes_per_thread=values["STACK"] + values["LOCAL"],
        stack_bytes=values["STACK"],
        static_local_bytes=values["LOCAL"],
    )


def validate_control(row, case):
    if row.get("role") != "control" or row.get("case") != case:
        raise ValueError("Require matching declared control")
    ptx = row["artifacts"]["ptx"]
    if hashlib.sha256(ptx.encode()).hexdigest() != row["artifact_sha256"]["ptx"]:
        raise ValueError("Control PTX fingerprint mismatch")
    target = re.search(r"^\.target\s+(sm_\d+a?)\s*$", ptx, re.M)
    if not target:
        raise ValueError("Unsupported PTX target")
    return ptx, target[1]


def compare_driver(row, declaration):
    keys = ("registers_per_thread", "local_bytes_per_thread")
    if row.get("launch_status") == "out_of_resources":
        if not row.get("launch_error") or any(row.get(k) is not None for k in keys):
            raise ValueError("Invalid unlaunchable control provenance")
        return dict(driver_comparable=False, driver_exact_match=None)
    if any(
        isinstance(row.get(k), bool) or not isinstance(row.get(k), int) or row[k] < 0
        for k in keys
    ):
        raise ValueError("Require real driver labels")
    return dict(
        driver_comparable=True,
        driver_exact_match=all(row[k] == declaration[k] for k in keys),
    )


def join_sources(source_root, resource_root, cubin_root):
    """Build an explicitly offline-label control table, retaining infeasible cases.

    Every binary must match its original control. All comparable driver labels
    must agree; unavailable driver labels remain unavailable in the raw data.
    This table is for allocation diagnostics, never kernel latency fitting.
    """
    source_root, resource_root, cubin_root = map(
        Path, (source_root, resource_root, cubin_root)
    )
    original_path = resource_root / "manifest.json"
    original = json.loads(original_path.read_text())
    sources = json.loads((source_root / "manifest.json").read_text())
    offline = json.loads((cubin_root / "manifest.json").read_text())
    if (
        any(m.get("role") != "control" for m in (original, sources, offline))
        or not original["cases"]
        or original["cases"] != sources["cases"]
        or original["cases"] != offline["cases"]
    ):
        raise ValueError("Require matching complete control declarations")
    if (
        original.get("packages", {}).get("triton") != "3.7.0"
        or hashlib.sha256(original_path.read_bytes()).hexdigest()
        != offline["source_manifest_sha256"]
    ):
        raise ValueError("Compiler version or manifest fingerprint mismatch")
    if len({c["id"] for c in original["cases"]}) != len(original["cases"]):
        raise ValueError("Duplicate controls")
    rows, compared = [], 0
    for case in original["cases"]:
        compiled = json.loads(
            (resource_root / "controls" / (case["id"] + ".json")).read_text()
        )
        source = json.loads(
            (source_root / "controls" / (case["id"] + ".json")).read_text()
        )
        ptx, _ = validate_control(compiled, case)
        if (
            source.get("role") != "control"
            or source["case"] != case
            or source.get("numerical_validation") != "passed"
            or source.get("compile_and_cuda_forbidden") is not True
        ):
            raise ValueError("Unverified source observation")
        folder = cubin_root / case["id"]
        result = json.loads((folder / "result.json").read_text())
        if (
            result.get("role") != "control"
            or result["case"] != case
            or result.get("eligible_for_fit") is not False
        ):
            raise ValueError("Invalid offline result identity")
        if (
            hashlib.sha256((folder / "control.cubin").read_bytes()).hexdigest()
            != compiled["cubin_sha256"]
            or result["cubin_sha256"] != compiled["cubin_sha256"]
            or result["ptx_sha256"] != hashlib.sha256(ptx.encode()).hexdigest()
            or (folder / "control.ptx").read_text() != ptx
        ):
            raise ValueError("Offline control artifact fingerprint mismatch")
        declaration = allocation_declaration(
            result["resource_usage"], kernel_name=compiled["kernel_name"]
        )
        if declaration != result["declaration"]:
            raise ValueError("Resource declaration differs from raw dump")
        comparison = compare_driver(compiled, declaration)
        if comparison["driver_exact_match"] is False:
            raise ValueError(
                "Offline/driver allocation mismatch; no label substitution"
            )
        compared += comparison["driver_comparable"]
        rows.append(
            {
                **source,
                **{
                    k: declaration[k]
                    for k in ("registers_per_thread", "local_bytes_per_thread")
                },
                "resource_label_provenance": "exact_control_cubin_allocation_declaration",
                "driver_comparable": comparison["driver_comparable"],
                "launch_status": compiled.get("launch_status", "loadable"),
            }
        )
    if not compared:
        raise ValueError("No independent driver comparisons")
    return dict(
        role="control",
        complete=True,
        compiler_version="3.7.0",
        rows=rows,
        eligible_for_fit=False,
        driver_compared=compared,
        caveat="Complete offline allocation labels, not complete feasible launches or latency labels.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ptxas", type=Path, required=True)
    parser.add_argument("--cuobjdump", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Require fresh output; preserve prior attempts")
    manifest_path = args.resource_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cases = manifest["cases"]
    if (
        manifest.get("role") != "control"
        or manifest.get("packages", {}).get("triton") != "3.7.0"
    ):
        raise ValueError("Require pinned control compiler declaration")
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Require nonempty unique controls")
    _write(
        args.output / "manifest.json",
        dict(
            role="control",
            cases=cases,
            source_manifest_sha256=hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            ptxas_sha256=hashlib.sha256(args.ptxas.read_bytes()).hexdigest(),
            cuobjdump_sha256=hashlib.sha256(args.cuobjdump.read_bytes()).hexdigest(),
            eligible_for_fit=False,
            gpu_execution=False,
        ),
    )
    results = []
    for case in cases:
        row = json.loads(
            (args.resource_root / "controls" / (case["id"] + ".json")).read_text()
        )
        ptx, target = validate_control(row, case)
        folder = args.output / case["id"]
        folder.mkdir()
        source, binary = folder / "control.ptx", folder / "control.cubin"
        source.write_text(ptx)
        command = [
            str(args.ptxas),
            "-lineinfo",
            "-v",
            "--regAllocOptLevel=2",
            "--gpu-name=" + target,
            str(source),
            "-o",
            str(binary),
        ]
        with (folder / "compile.log").open("x") as log:
            subprocess.run(
                command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=300
            )
        digest = hashlib.sha256(binary.read_bytes()).hexdigest()
        if digest != row["cubin_sha256"]:
            raise ValueError("Recompiled cubin differs from archived control")
        usage = subprocess.run(
            [str(args.cuobjdump), "--dump-resource-usage", str(binary)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        declaration = allocation_declaration(usage, kernel_name=row["kernel_name"])
        result = dict(
            role="control",
            case=case,
            cubin_sha256=digest,
            ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(),
            command=command,
            resource_usage=usage,
            declaration=declaration,
            **compare_driver(row, declaration),
            eligible_for_fit=False,
        )
        _write(folder / "result.json", result)
        results.append(result)
        print(
            case["id"],
            result["driver_comparable"],
            result["driver_exact_match"],
            flush=True,
        )
    _write(
        args.output / "summary.json",
        dict(
            role="control",
            count=len(results),
            compared=sum(r["driver_comparable"] for r in results),
            exact=sum(r["driver_exact_match"] is True for r in results),
            eligible_for_fit=False,
            rows=results,
            caveat="Offline allocation declarations, not latency labels or feasibility of a launch. No target artifacts or latency CV.",
        ),
    )


if __name__ == "__main__":
    main()
