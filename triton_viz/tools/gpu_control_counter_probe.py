"""Profile one declared control's dynamic counters, never fit profiler timing."""

import argparse
import ctypes
import os

from triton_viz.tools.gpu_control_resources import selected_controls


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=(
            "geometry",
            "structure",
            "pressure",
            "pressure_pipeline",
            "resource_dot",
        ),
        required=True,
    )
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--allow-idle-graphics", action="store_true")
    args = parser.parse_args(argv)
    cases = [
        case for case in selected_controls(args.suite) if case["id"] == args.case_id
    ]
    if len(cases) != 1:
        raise ValueError("Case must belong to the declared control suite")
    case = cases[0]

    import torch
    import triton_viz
    from microbench.gpu.harness.measure import assert_available, snapshot
    from microbench.gpu.tests.coverage.kernels import prepare, check_output
    from triton_viz.performance.triton_observe import observe

    baseline = snapshot(0)
    graphics = (
        tuple(p["pid"] for p in baseline["graphics_processes"])
        if args.allow_idle_graphics
        else ()
    )
    assert_available(baseline, allowed_graphics=graphics)
    options = {
        key: case.get(key, value)
        for key, value in (("num_warps", 4), ("num_stages", 2))
    }
    kernel, grid, inputs, out = prepare(case, "cpu")
    observe(kernel, grid, *inputs, **options)
    check_output(case, out)
    triton_viz.clear()
    kernel, grid, inputs, out = prepare(case, "cuda:0")
    kernel[grid](*inputs, **options)
    torch.cuda.synchronize()
    assert_available(snapshot(0), own_pid=os.getpid(), allowed_graphics=graphics)
    driver = ctypes.CDLL("libcuda.so.1")
    if driver.cuProfilerStart():
        raise RuntimeError("Cannot start control profiling")
    kernel[grid](*inputs, **options)
    torch.cuda.synchronize()
    if driver.cuProfilerStop():
        raise RuntimeError("Cannot stop control profiling")
    check_output(case, out)
    print(
        f"control={case['id']} numerical=passed profiler_timing_not_for_fit", flush=True
    )


if __name__ == "__main__":
    main()
