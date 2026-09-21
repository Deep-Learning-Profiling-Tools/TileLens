"""Summarize complete declared control compiler artifacts, without timings."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from triton_viz.tools.gpu_cost_model_pipeline import _write


def decode_local_allocation(row, *, compiler_version):
    """Decode the pinned NVIDIA driver resource field, not spill traffic.

    Triton 3.7 driver.c queries CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES and performs
    integer division by four before returning n_spills. Our explicit-byte
    collector also multiplies that returned field by four. Both paths therefore
    encode the SAME four-byte-quantized quantity, not independent measurements.
    """
    if compiler_version != "3.7.0":
        raise ValueError("Unverified resource-field semantics for compiler version")
    units = row.get("triton_reported_spills")
    if isinstance(units, bool) or not isinstance(units, int) or units < 0:
        raise ValueError("Require nonnegative integer driver local-allocation units")
    value = 4 * units
    if "local_bytes_per_thread" in row:
        explicit = row["local_bytes_per_thread"]
        if (
            isinstance(explicit, bool)
            or not isinstance(explicit, int)
            or explicit != value
        ):
            raise ValueError(
                "Conflicting explicit and driver-derived allocation labels"
            )
    return dict(
        local_bytes_per_thread=value,
        quantization_bytes=4,
        provenance="triton_3.7_nvidia_driver_LOCAL_SIZE_BYTES_div4_times4",
    )


def blocked_dot_fragments(ttgir):
    """Expose register-fragment replication in explicit blocked dot layouts.

    Fully materialized operand fragments are NOT peak allocated registers:
    LLVM/ptxas scheduling can stream, reuse, or spill them. Tensor-core layouts
    stay unsupported here instead of being approximated by this SIMT mapping.
    """
    layouts = {}
    for alias, body in re.findall(
        r"^(#[\w]+) = #ttg\.blocked<\{([^\n]+)\}>", ttgir, re.M
    ):
        fields = {}
        for name in ("sizePerThread", "threadsPerWarp", "warpsPerCTA"):
            match = re.search(name + r" = \[(\d+), (\d+)\]", body)
            if match:
                fields[name] = [int(v) for v in match.groups()]
        if len(fields) == 3 and all(v > 0 for pair in fields.values() for v in pair):
            order = re.search(r"order = \[(\d+), (\d+)\]", body)
            if order and sorted(int(v) for v in order.groups()) == [0, 1]:
                fields["order"] = [int(v) for v in order.groups()]
            layouts[alias] = fields
    rows = []
    for line in ttgir.splitlines():
        if not re.search(r"\btt\.dot\b", line):
            continue
        shapes = re.findall(r"tensor<(\d+)x(\d+)x(f32|f16|bf16),", line)
        output = re.search(r"-> tensor<\d+x\d+xf32, (#[\w]+)>", line)
        if (
            len(shapes) != 3
            or not output
            or output[1] not in layouts
            or any(s[2] != "f32" for s in shapes)
        ):
            rows.append(
                dict(
                    supported=False, reason="non-blocked-FP32 or unsupported dot layout"
                )
            )
            continue
        (m, k, _), (kb, n, _), (mo, no, _) = shapes
        m, n, k = int(m), int(n), int(k)
        if (int(kb), int(mo), int(no)) != (k, m, n):
            raise ValueError("Inconsistent control dot geometry")
        layout = layouts[output[1]]
        sizes = layout["sizePerThread"]
        coverage = [
            sizes[i] * layout["threadsPerWarp"][i] * layout["warpsPerCTA"][i]
            for i in range(2)
        ]
        fragments = [
            sizes[i] * ((shape + coverage[i] - 1) // coverage[i])
            for i, shape in enumerate((m, n))
        ]
        rows.append(
            dict(
                supported=True,
                shape=[m, n, k],
                layout=layout,
                accumulator_words_per_thread=fragments[0] * fragments[1],
                fully_materialized_operand_words_per_thread=k * sum(fragments),
            )
        )
    return rows


def sass_backedges(sass):
    """Report direct backward branch regions, excluding terminal self spins.

    These are static CFG observations, not inferred execution counts. Region
    overlap or predication requires additional analysis before dynamic pricing.
    """
    instructions = []
    for line in sass.splitlines():
        match = re.search(
            r"/\*([0-9a-fA-F]+)\*/\s+(?:@!?U?P(?:[0-9]+|T)\s+)?([A-Z][A-Z0-9]*(?:\.[A-Z0-9]+)*)\s+([^;]*);",
            line,
        )
        if match:
            pc, opcode, operands = match.groups()
            instructions.append((int(pc, 16), opcode.split(".", 1)[0], operands))
    if len({pc for pc, _, _ in instructions}) != len(instructions):
        return dict(
            supported=False, reason="multiple functions or duplicate PCs", regions=[]
        )
    regions = []
    for pc, opcode, operands in instructions:
        target = re.search(r"\b0x([0-9a-fA-F]+)\b", operands)
        if opcode == "BRA" and target and int(target[1], 16) < pc:
            start = int(target[1], 16)
            regions.append(
                dict(
                    start_pc=start,
                    branch_pc=pc,
                    static_counts=dict(
                        Counter(
                            op for addr, op, _ in instructions if start <= addr <= pc
                        )
                    ),
                )
            )
    return dict(supported=True, regions=regions)


def audit(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("role") != "control":
        raise ValueError("Resource audit accepts only a control manifest")
    rows = []
    for case in manifest["cases"]:
        path = root / "controls" / (case["id"] + ".json")
        if not path.exists():
            rows.append(dict(case=case, status="missing"))
            continue
        row = json.loads(path.read_text())
        if row.get("role") != "control" or row["case"] != case:
            raise ValueError("Control artifact identity mismatch")
        for name, artifact in row["artifacts"].items():
            if (
                hashlib.sha256(artifact.encode()).hexdigest()
                != row["artifact_sha256"][name]
            ):
                raise ValueError("Compiler artifact digest mismatch")
        ptx = row["artifacts"].get("ptx", "")
        sass = row["artifacts"].get("sass")
        sass_counts = (
            None
            if sass is None
            else dict(
                Counter(
                    match.split(".", 1)[0]
                    for match in re.findall(
                        r"/\*[0-9a-fA-F]+\*/\s+(?:@!?U?P(?:[0-9]+|T)\s+)?([A-Z][A-Z0-9]*(?:\.[A-Z0-9]+)*)",
                        sass,
                    )
                )
            )
        )
        # Static mnemonics indicate lowering choices, not dynamic instruction
        # counts. In particular, ptxas can introduce spills after PTX lowering.
        patterns = {
            "mma": r"\b(?:mma|wgmma|tcgen05\.mma)\.",
            "async_copy": r"\bcp\.async\.",
            "barrier": r"\b(?:bar\.sync|mbarrier\.)",
            "local_load": r"\bld\.local\.",
            "local_store": r"\bst\.local\.",
            "fp32_fma": r"\bfma\.rn\.f32\b",
            "fp32_fma_x2": r"\bfma\.rn\.f32x2\b",
        }
        rows.append(
            dict(
                case=case,
                status="complete",
                registers_per_thread=row["registers_per_thread"],
                triton_reported_spills=row["triton_reported_spills"],
                local_bytes_per_thread=row.get(
                    "local_bytes_per_thread", 4 * row["triton_reported_spills"]
                ),
                shared_bytes=row["shared_bytes"],
                blocked_dot_fragments=blocked_dot_fragments(
                    row["artifacts"].get("ttgir", "")
                ),
                static_ttgir_loop_count=len(
                    re.findall(r"\bscf\.for\b", row["artifacts"].get("ttgir", ""))
                ),
                static_sass_counts=sass_counts,
                sass_backedges=None if sass is None else sass_backedges(sass),
                static_ptx_counts={
                    name: len(re.findall(pattern, ptx))
                    for name, pattern in patterns.items()
                },
                static_sass_local_counts=None
                if sass is None
                else {
                    "loads": len(re.findall(r"\bLDL(?:\.|\s)", sass)),
                    "stores": len(re.findall(r"\bSTL(?:\.|\s)", sass)),
                },
            )
        )
    return dict(
        role="control",
        count=len(rows),
        complete=all(row["status"] == "complete" for row in rows),
        rows=rows,
        caveat="Static control resource diagnostics only; no occupancy/spill latency model is fitted.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Use a new audit output")
    result = audit(args.root)
    _write(args.output, result)
    print({key: result[key] for key in ("count", "complete")})


if __name__ == "__main__":
    main()
