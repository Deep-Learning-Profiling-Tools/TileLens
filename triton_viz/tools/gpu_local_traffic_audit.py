"""Control-only typed local-traffic accounting, not a source-only predictor.

Explicit loop trips are an audited hypothesis, never inferred from counters.
Payload sector equivalents assume full active lanes and contiguous local words;
they are not DRAM misses or proof of a general local-memory transaction model.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

from triton_viz.tools.gpu_control_resource_audit import sass_backedges
from triton_viz.tools.gpu_cost_model_pipeline import _write


def account(row, *, loop_trips):
    if row.get("role") != "control":
        raise ValueError("Only control compiler artifacts may be audited")
    if (
        isinstance(loop_trips, bool)
        or not isinstance(loop_trips, int)
        or loop_trips < 1
    ):
        raise ValueError("Require positive explicit loop trips")
    sass = row["artifacts"]["sass"]
    if hashlib.sha256(sass.encode()).hexdigest() != row["artifact_sha256"]["sass"]:
        raise ValueError("SASS fingerprint mismatch")
    cfg = sass_backedges(sass)
    if not cfg["supported"] or len(cfg["regions"]) > 1:
        raise ValueError("Require at most one direct loop region")
    region = cfg["regions"][0] if cfg["regions"] else None
    if region is None and loop_trips != 1:
        raise ValueError("Straight-line SASS executes once, not source repeat times")
    totals = {scope: {op: 0 for op in ("LDL", "STL")} for scope in ("loop", "outside")}
    for line in sass.splitlines():
        match = re.search(
            r"/\*([0-9a-fA-F]+)\*/\s+(?:(@!?U?P\w+)\s+)?"
            r"([A-Z][A-Z0-9]*(?:\.[A-Z0-9]+)*)\s+([^;]*);",
            line,
        )
        if not match:
            continue
        address, predicate, opcode, operands = match.groups()
        pc = int(address, 16)
        base = opcode.split(".")[0]
        if base in {"CALL", "RET", "BRX", "JMX", "JMP"}:
            raise ValueError("Unsupported indirect or interprocedural control flow")
        if base == "BRA":
            target = re.search(r"\b0x([0-9a-fA-F]+)\b", operands)
            if not target or (
                (region is None or pc != region["branch_pc"])
                and int(target[1], 16) != pc
            ):
                raise ValueError("Additional branch requires path-sensitive accounting")
        if base not in {"LDL", "STL"}:
            continue
        if predicate:
            raise ValueError("Predicated local accesses require active-lane evidence")
        suffixes = opcode.split(".")[1:]
        if set(suffixes) - {"LU", "64", "128"} or {"64", "128"} <= set(suffixes):
            raise ValueError("Unsupported local-memory width/modifier")
        width = 16 if "128" in suffixes else 8 if "64" in suffixes else 4
        scope = (
            "loop"
            if region is not None and region["start_pc"] <= pc <= region["branch_pc"]
            else "outside"
        )
        totals[scope][base] += width
    case = row["case"]
    programs, warps = case["programs"], case["num_warps"]
    if any(
        isinstance(x, bool) or not isinstance(x, int) or x <= 0
        for x in (programs, warps)
    ):
        raise ValueError("Require explicit positive program and warp counts")
    dynamic = {
        op: totals["outside"][op] + loop_trips * totals["loop"][op]
        for op in ("LDL", "STL")
    }
    return dict(
        role="control",
        case=case,
        sass_sha256=row["artifact_sha256"]["sass"],
        loop_region=region,
        assumed_loop_trips=loop_trips,
        static_bytes_per_thread=totals,
        conditional_bytes_per_thread=dynamic,
        conditional_payload_sector_equivalents={
            op: value * programs * warps for op, value in dynamic.items()
        },
        eligible_for_fit=False,
        caveat="Conditional full-lane payload accounting; validate loop trips, local layout and counters independently. Not source-only prediction or DRAM traffic.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--loop-trips", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Use a fresh audit output")
    _write(
        args.output,
        account(json.loads(args.control.read_text()), loop_trips=args.loop_trips),
    )


if __name__ == "__main__":
    main()
