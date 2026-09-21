"""Numerically validate an archived control intervention; never collect timing."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

from triton_viz.tools.gpu_cost_model_pipeline import _write


def profile_one_variant(driver, launch, synchronize):
    """Bracket one precompiled launch; no timer or fallback is used."""
    if driver.cuProfilerStart():
        raise RuntimeError("Cannot start control counter range")
    try:
        launch()
        synchronize()
    finally:
        if driver.cuProfilerStop():
            raise RuntimeError("Cannot stop control counter range")


def clone_with_binary(compiled, record, binary, *, compiler_version):
    """Pinned Triton loader adaptation; no mutation of the cached original."""
    if compiler_version != "3.7.0":
        raise ValueError("Require verified Triton 3.7.0 loader contract")
    if record.get("role") != "control" or record.get("eligible_for_fit") is not False:
        raise ValueError("Require a diagnostic control record")
    if record.get("case", {}).get("kind") != "geometry_dot":
        raise ValueError("Require pure-dot controls")
    if hashlib.sha256(binary).hexdigest() != record["cubin_sha256"]:
        raise ValueError("Intervention cubin fingerprint mismatch")
    if hashlib.sha256(compiled.kernel).hexdigest() != record["archived_cubin_sha256"]:
        raise ValueError("Recompiled original differs from archived baseline")
    if compiled.name != "geometry_dot":
        raise ValueError("Unexpected control entry point")
    result = copy.copy(compiled)
    result.asm = dict(compiled.asm, cubin=binary)
    result.kernel = binary
    result.hash = record["cubin_sha256"]
    result.module = result.function = result._run = None
    for name in ("n_regs", "n_spills", "n_max_threads"):
        if hasattr(result, name):
            delattr(result, name)
    return result


def load_control(resource_root, intervention_root, identity, variant):
    from triton_viz.tools.gpu_packing_intervention import variants

    manifest = json.loads((intervention_root / "manifest.json").read_text())
    sources = json.loads((resource_root / "manifest.json").read_text())
    if manifest.get("role") != "control" or sources.get("role") != "control":
        raise ValueError("Require control manifests")
    selected = [c for c in manifest["cases"] if c["id"] == identity]
    if len(selected) != 1 or selected[0] not in sources["cases"]:
        raise ValueError("Control declaration mismatch")
    raw = json.loads((resource_root / "controls" / (identity + ".json")).read_text())
    if raw["case"] != selected[0]:
        raise ValueError("Resource control identity mismatch")
    if variant not in {"nounroll", "sync_copy"} or manifest["intervention"] != variant:
        raise ValueError("Unsupported or mismatched control intervention")
    path = intervention_root / identity / variant
    record = json.loads(path.with_suffix(".json").read_text())
    expected, _ = variants(raw, variant)
    ptx = path.with_suffix(".ptx").read_text()
    if (
        record["case"] != selected[0]
        or record["variant"] != variant
        or record["archived_cubin_sha256"] != raw["cubin_sha256"]
        or ptx != expected[variant]
        or hashlib.sha256(ptx.encode()).hexdigest() != record["ptx_sha256"]
    ):
        raise ValueError("Intervention provenance differs from control transformation")
    binary = path.with_suffix(".cubin").read_bytes()
    if (
        record.get("role") != "control"
        or hashlib.sha256(binary).hexdigest() != record["cubin_sha256"]
    ):
        raise ValueError("Unverified control binary")
    return selected[0], record, binary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--intervention-root", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--variant", choices=("nounroll", "sync_copy"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-idle-graphics", action="store_true")
    parser.add_argument(
        "--profile-variant",
        action="store_true",
        help="Bracket one additional validated variant launch for external counters; no timing",
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Require a fresh verification record")
    case, record, binary = load_control(
        args.resource_root, args.intervention_root, args.case_id, args.variant
    )
    import torch
    import triton
    import triton_viz
    from microbench.gpu.harness.measure import assert_available, snapshot
    from microbench.gpu.tests.coverage.kernels import prepare, check_output
    from triton_viz.performance.triton_observe import observe

    if triton.__version__ != "3.7.0":
        raise ValueError("Require verified Triton 3.7.0 loader contract")
    baseline = snapshot(0)
    graphics = (
        tuple(p["pid"] for p in baseline["graphics_processes"])
        if args.allow_idle_graphics
        else ()
    )
    assert_available(baseline, allowed_graphics=graphics)
    options = {k: case[k] for k in ("num_warps", "num_stages")}
    kernel, grid, inputs, out = prepare(case, "cpu")
    observe(kernel, grid, *inputs, **options)
    check_output(case, out)
    triton_viz.clear()
    kernel, grid, inputs, out = prepare(case, "cuda:0")
    original = kernel.warmup(*inputs, grid=grid, **options)
    mutated = clone_with_binary(
        original, record, binary, compiler_version=triton.__version__
    )
    grid3 = tuple(grid) + (1,) * (3 - len(grid))
    # Initialize both module handles before validation launches, without timing.
    original._init_handles()
    mutated._init_handles()
    for compiled in (original, mutated):
        out.fill_(float("nan"))
        torch.cuda.synchronize()
        assert_available(snapshot(0), own_pid=os.getpid(), allowed_graphics=graphics)
        compiled[grid3](*inputs)
        torch.cuda.synchronize()
        assert_available(snapshot(0), own_pid=os.getpid(), allowed_graphics=graphics)
        check_output(case, out)
    if args.profile_variant:
        import ctypes

        out.fill_(float("nan"))
        torch.cuda.synchronize()
        assert_available(snapshot(0), own_pid=os.getpid(), allowed_graphics=graphics)
        profile_one_variant(
            ctypes.CDLL("libcuda.so.1"),
            lambda: mutated[grid3](*inputs),
            torch.cuda.synchronize,
        )
        assert_available(snapshot(0), own_pid=os.getpid(), allowed_graphics=graphics)
        check_output(case, out)
    _write(
        args.output,
        dict(
            role="control",
            case=case,
            variant=args.variant,
            cubin_sha256=record["cubin_sha256"],
            archived_cubin_sha256=record["archived_cubin_sha256"],
            numerical_validation="passed",
            eligible_for_fit=False,
            timing_collected=False,
            profiled_variant_launch=args.profile_variant,
            caveat="Declared deterministic control inputs only. No counter or latency validation; snapshots are not continuous interference monitoring.",
        ),
    )
    print(
        f"control={case['id']} variant={args.variant} numerical=passed no_timing",
        flush=True,
    )


if __name__ == "__main__":
    main()
