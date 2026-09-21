"""Summarize complete declared control compiler artifacts, without timings."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from triton_viz.tools.gpu_cost_model_pipeline import _write


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
        # Static mnemonics indicate lowering choices, not dynamic instruction
        # counts. In particular, ptxas can introduce spills after PTX lowering.
        patterns = {
            "mma": r"\b(?:mma|wgmma|tcgen05\.mma)\.",
            "async_copy": r"\bcp\.async\.",
            "barrier": r"\b(?:bar\.sync|mbarrier\.)",
            "local_load": r"\bld\.local\.",
            "local_store": r"\bst\.local\.",
            "fp32_fma": r"\bfma\.rn\.f32\b",
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
