"""Declared control-only cache experiment; run under ncu application replay.

No timing from this profiler run may enter latency calibration. Use ncu with
--profile-from-start off --cache-control none --clock-control none and preserve
the complete application setup on every replay pass.
"""

from __future__ import annotations

import argparse
import ctypes
import os

import triton
import triton.language as tl


@triton.jit
def cache_read_control(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # Bypass L1 to isolate L2 reuse and eviction in this control.
    value = tl.load(X + offsets, offsets < N, other=0, cache_modifier=".cg")
    tl.store(Y + offsets, value + 1, offsets < N)


@triton.jit
def cache_eviction_read(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + offsets, offsets < N, other=0, cache_modifier=".cg")
    tl.store(Y + offsets, value + 1, offsets < N)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--working-set-mib", type=int, choices=(3, 12, 48), required=True
    )
    parser.add_argument("--eviction", choices=("none", "zero", "read"), required=True)
    parser.add_argument(
        "--block",
        type=int,
        choices=(1024, 4096),
        default=1024,
        help="Diagnostic launch geometry; working set is unchanged",
    )
    parser.add_argument("--allow-idle-graphics", action="store_true")
    parser.add_argument(
        "--synchronize-launches",
        action="store_true",
        help="Diagnostic variant: host-sync between priming launches",
    )
    args = parser.parse_args(argv)

    import torch
    from microbench.gpu.harness.measure import assert_available, snapshot

    baseline = snapshot(0)
    graphics = (
        tuple(p["pid"] for p in baseline["graphics_processes"])
        if args.allow_idle_graphics
        else ()
    )
    assert_available(baseline, allowed_graphics=graphics)
    n = args.working_set_mib * 1024 * 1024 // 4
    block = args.block
    x = torch.arange(n, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    driver = ctypes.CDLL("libcuda.so.1")
    l2 = ctypes.c_int()
    if driver.cuDeviceGetAttribute(ctypes.byref(l2), 38, 0) or l2.value <= 0:
        raise RuntimeError("Cannot query L2 capacity")
    sweep = torch.empty(2 * l2.value, dtype=torch.uint8, device="cuda")
    sweep_out = torch.empty_like(sweep) if args.eviction == "read" else None
    # Compile all control launches before constructing the measured cache state.
    cache_read_control[(triton.cdiv(n, block),)](x, y, n, block)
    if sweep_out is not None:
        sweep.zero_()
        cache_eviction_read[(triton.cdiv(sweep.numel(), 1024),)](
            sweep, sweep_out, sweep.numel(), 1024
        )
    torch.cuda.synchronize()
    assert_available(snapshot(0), own_pid=os.getpid(), allowed_graphics=graphics)
    # Collect all three cache_read_control launches, including both priming
    # launches. On this host the first *profiled* launch is cold even when
    # unprofiled launches prime it. Skipping priming in ncu therefore destroys
    # the warm positive control. Earlier failed protocols remain archived.
    if driver.cuProfilerStart():
        raise RuntimeError("Cannot start control counter capture")
    for _ in range(2):
        cache_read_control[(triton.cdiv(n, block),)](x, y, n, block)
        if args.synchronize_launches:
            torch.cuda.synchronize()
    if args.eviction == "zero":
        sweep.zero_()
    elif args.eviction == "read":
        cache_eviction_read[(triton.cdiv(sweep.numel(), 1024),)](
            sweep, sweep_out, sweep.numel(), 1024
        )
    cache_read_control[(triton.cdiv(n, block),)](x, y, n, block)
    torch.cuda.synchronize()
    if driver.cuProfilerStop():
        raise RuntimeError("Cannot stop control counter capture")
    torch.testing.assert_close(y, x + 1, rtol=0, atol=0)
    print(
        f"control cache_read: {args.working_set_mib} MiB, eviction={args.eviction}, block={block}, L2={l2.value}, numerical=passed",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
