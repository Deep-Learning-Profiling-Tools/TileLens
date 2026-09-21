"""Archive compiler resources of declared controls, never target kernels.

Resource reports are diagnostic training data, not prediction inputs. This
tool does not produce a model or read any measurement/holdout directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
import subprocess
import tempfile
from pathlib import Path

from microbench.gpu.common.cases import load_cases
from triton_viz.tools.gpu_cost_model_pipeline import _write


def selected_controls(suite):
    if suite not in {
        "geometry",
        "structure",
        "pressure",
        "pressure_pipeline",
        "resource_transfer",
        "composition_component",
    }:
        raise ValueError(
            "Only declared geometry/structure/pressure/resource_transfer controls are supported"
        )
    cases = load_cases(suite, "control")
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate control identifier")
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=(
            "geometry",
            "structure",
            "pressure",
            "pressure_pipeline",
            "resource_transfer",
            "composition_component",
        ),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-idle-graphics", action="store_true")
    parser.add_argument(
        "--cuobjdump", type=Path, help="Optional control-only SASS disassembly"
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Use a new artifact root; preserve prior attempts")
    cases = selected_controls(args.suite)
    from microbench.gpu.harness.measure import assert_available, snapshot

    baseline = snapshot(0)
    graphics = (
        tuple(p["pid"] for p in baseline["graphics_processes"])
        if args.allow_idle_graphics
        else ()
    )
    assert_available(baseline, allowed_graphics=graphics)
    _write(
        args.output / "manifest.json",
        dict(
            role="control",
            suite=args.suite,
            cases=cases,
            hardware=baseline,
            packages={
                name: importlib.metadata.version(name) for name in ("torch", "triton")
            },
            compiler_cache=os.environ.get("TRITON_CACHE_DIR"),
        ),
    )
    from microbench.gpu.tests.coverage.kernels import prepare

    for case in cases:
        assert_available(snapshot(0), own_pid=os.getpid(), allowed_graphics=graphics)
        kernel, grid, inputs, out = prepare(case, "cuda:0")
        options = {
            key: case.get(key, value)
            for key, value in (("num_warps", 4), ("num_stages", 2))
        }
        compiled = kernel.warmup(*inputs, grid=grid, **options)
        compiled._init_handles()
        artifacts = {
            name: compiled.asm[name]
            for name in ("ptx", "ttgir", "llir")
            if name in compiled.asm
        }
        if args.cuobjdump is not None:
            with tempfile.TemporaryDirectory(prefix="gpu-control-cubin-") as scratch:
                binary = Path(scratch) / "control.cubin"
                binary.write_bytes(compiled.asm["cubin"])
                artifacts["sass"] = subprocess.run(
                    [
                        str(args.cuobjdump.resolve(strict=True)),
                        "--dump-sass",
                        str(binary),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                ).stdout
        record = dict(
            role="control",
            case=case,
            kernel_name=compiled.name,
            registers_per_thread=compiled.n_regs,
            triton_reported_spills=compiled.n_spills,
            local_bytes_per_thread=4 * compiled.n_spills,
            shared_bytes=compiled.metadata.shared,
            num_warps=compiled.metadata.num_warps,
            artifacts=artifacts,
            artifact_sha256={
                k: hashlib.sha256(v.encode()).hexdigest() for k, v in artifacts.items()
            },
            cubin_sha256=hashlib.sha256(compiled.asm["cubin"]).hexdigest(),
            note="Triton n_spills is CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES / 4, not a dynamic spill count; validate actual traffic with control SASS/counters.",
        )
        _write(args.output / "controls" / (case["id"] + ".json"), record)
        print(
            case["id"],
            record["registers_per_thread"],
            record["triton_reported_spills"],
            record["shared_bytes"],
            flush=True,
        )
        del inputs, out
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
