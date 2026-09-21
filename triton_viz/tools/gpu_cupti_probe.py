"""Control-only CUPTI/eviction smoke test, never a calibration dataset.

The eviction sweep must still be checked with separate control-only cache
counters. A successful probe does not certify cold-cache measurements.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import statistics
import math
import time
from pathlib import Path

from microbench.gpu.harness.cupti import (
    CuptiTimestamps,
    validate_timestamps,
    group_kernel_intervals,
)
from microbench.gpu.harness.measure import assert_available, snapshot, monitor_call
from triton_viz.tools.gpu_cost_model_pipeline import _write


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-idle-graphics", action="store_true")
    parser.add_argument(
        "--suite",
        choices=(
            "coverage",
            "pressure",
            "pressure_pipeline",
            "resource_transfer",
            "composition_component",
            "geometry",
            "structure",
            "stability",
        ),
        default="coverage",
    )
    parser.add_argument(
        "--monitored",
        action="store_true",
        help="Retain continuous NVML process monitoring around the measured graph batch",
    )
    parser.add_argument("--warmup-seconds", type=float, default=0.5)
    parser.add_argument("--kernels-per-sample", type=int, default=1)
    parser.add_argument(
        "--graph-pairs-per-replay",
        type=int,
        help="Bound graph size and drain every replay; separate diagnostic protocol",
    )
    parser.add_argument("--delivery-timeout-seconds", type=float, default=1.0)
    parser.add_argument(
        "--timestamp-method", choices=("hes", "software_serial"), default="hes"
    )
    parser.add_argument(
        "--eviction-mode", choices=("torch_zero", "persistent_sm"), default="torch_zero"
    )
    parser.add_argument(
        "--capture-cache",
        action="store_true",
        help="Retain anonymous source-sector accesses for control-only cache modeling",
    )
    parser.add_argument(
        "--graph-samples",
        action="store_true",
        help="Capture all eviction/control pairs; time each kernel via HES",
    )
    parser.add_argument(
        "--graph-debug",
        action="store_true",
        help="Save the declared control's CUDA graph structure for collector diagnostics",
    )
    parser.add_argument(
        "--case-id", help="Exact declared control identifier; never a target path"
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.warmup_seconds) or args.warmup_seconds < 0:
        raise ValueError("Warmup duration must be finite and nonnegative")
    if args.output.exists():
        raise ValueError("Use a fresh probe output; do not overwrite attempts")
    if args.kernels_per_sample < 1:
        raise ValueError("Replicated samples require a positive count")
    if args.graph_pairs_per_replay is not None and (
        not args.graph_samples
        or args.graph_pairs_per_replay < 1
        or (11 * args.kernels_per_sample) % args.graph_pairs_per_replay
    ):
        raise ValueError("Graph chunk must divide the full declared launch count")
    baseline = snapshot(0)
    graphics = (
        tuple(p["pid"] for p in baseline["graphics_processes"])
        if args.allow_idle_graphics
        else ()
    )
    assert_available(baseline, allowed_graphics=graphics)
    collector = CuptiTimestamps(
        args.library,
        delivery_timeout_seconds=args.delivery_timeout_seconds,
        timestamp_method=args.timestamp_method,
    )  # Before any CUDA context.
    try:
        return _run_probe(args, baseline, graphics, collector)
    except Exception as error:
        failure = dict(role="control", eligible_for_fit=False, failed=str(error))
        try:
            failure["records"] = collector.read()
        except Exception as delivery_error:
            failure["record_delivery_error"] = str(delivery_error)
        failure_path = args.output.with_suffix(".failure.json")
        if not failure_path.exists():
            _write(failure_path, failure)
        raise
    finally:
        collector.close()
        failure_path = args.output.with_suffix(".failure.json")
        teardown_path = args.output.with_suffix(".teardown.json")
        if failure_path.exists() and not teardown_path.exists():
            _write(
                teardown_path,
                dict(
                    role="control",
                    eligible_for_fit=False,
                    diagnostic="post-finalize forced delivery; may contain incomplete records",
                    records=collector.snapshot(),
                ),
            )


def _run_probe(args, baseline, graphics, collector):
    import os
    import torch
    import triton_viz
    from microbench.gpu.common.cases import load_cases
    from microbench.gpu.tests.coverage.kernels import prepare, check_output
    from triton_viz.performance.triton_observe import observe

    controls = load_cases(args.suite, "control")
    if args.case_id:
        matches = [c for c in controls if c["id"] == args.case_id]
        if len(matches) != 1:
            raise ValueError("Unknown declared control identifier")
        case = matches[0]
    elif args.suite == "coverage":
        case = next(c for c in controls if c["kind"] == "coverage_vector")
    else:
        raise ValueError("Pressure probes require an explicit declared case ID")
    kernel, grid, inputs, out = prepare(case, "cpu")
    options = {
        key: case.get(key, default)
        for key, default in (("num_warps", 4), ("num_stages", 2))
    }
    source = observe(kernel, grid, *inputs, capture_cache=args.capture_cache, **options)
    check_output(case, out)
    triton_viz.clear()
    driver = ctypes.CDLL("libcuda.so.1")
    l2_bytes = ctypes.c_int()
    # CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE from cuda.h (bytes).
    if (
        driver.cuDeviceGetAttribute(ctypes.byref(l2_bytes), 38, 0)
        or l2_bytes.value <= 0
    ):
        raise RuntimeError("Cannot query L2 capacity")
    sweep_bytes = 2 * l2_bytes.value
    sweep = torch.empty(sweep_bytes, dtype=torch.uint8, device="cuda:0")
    if args.eviction_mode == "persistent_sm":
        from microbench.gpu.harness.eviction import persistent_eviction

        sm_count = torch.cuda.get_device_properties(0).multi_processor_count

        def evict():
            persistent_eviction[(sm_count,)](
                sweep, sweep_bytes, BLOCK=4096, PROGRAMS=sm_count, num_warps=4
            )
    else:
        evict = sweep.zero_
    kernel, grid, inputs, out = prepare(case, "cuda:0")

    def launch():
        kernel[grid](*inputs, **options)

    # Compile and warm up outside measured samples; infer names only from this
    # declared control, not from a target compiler cache.
    evict()
    launch()
    torch.cuda.synchronize()
    collector.read(expected_count=2)
    check_output(case, out)
    collector.clear()
    evict()
    launch()
    torch.cuda.synchronize()
    warm_records = sorted(collector.read(expected_count=2), key=lambda r: r["start_ns"])
    if len(warm_records) != 2:
        _write(
            args.output,
            dict(
                role="control",
                case=case,
                eligible_for_fit=False,
                failed="unexpected_warmup_records",
                records=warm_records,
            ),
        )
        raise RuntimeError(
            f"Expected exactly one eviction and one control kernel, got {warm_records}"
        )
    names = [r["name"] for r in warm_records]
    if names[0] == names[1]:
        raise RuntimeError("Cannot distinguish eviction from target activity")
    deadline = time.monotonic() + args.warmup_seconds
    warmup_launches = 0
    collector.clear()
    while time.monotonic() < deadline:
        evict()
        launch()
        torch.cuda.synchronize()
        collector.read(expected_count=2)
        collector.clear()
        warmup_launches += 1
    samples = []
    monitoring = None
    telemetry = [baseline]
    if args.graph_samples:
        launches = 11 * args.kernels_per_sample
        pairs_per_replay = args.graph_pairs_per_replay or launches
        expected_nodes = 2 * pairs_per_replay
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            for _ in range(pairs_per_replay):
                evict()
                launch()
        handle = ctypes.c_void_p(graph.raw_cuda_graph())
        node_count = ctypes.c_size_t()
        if driver.cuGraphGetNodes(handle, None, ctypes.byref(node_count)):
            raise RuntimeError("Cannot inspect declared control graph")
        nodes = (ctypes.c_void_p * node_count.value)()
        if driver.cuGraphGetNodes(handle, nodes, ctypes.byref(node_count)):
            raise RuntimeError("Cannot retrieve control graph nodes")
        graph_kernel_nodes = 0
        for node in nodes:
            kind = ctypes.c_int()
            if driver.cuGraphNodeGetType(ctypes.c_void_p(node), ctypes.byref(kind)):
                raise RuntimeError("Cannot inspect graph node kind")
            graph_kernel_nodes += kind.value == 0  # CU_GRAPH_NODE_TYPE_KERNEL
        if graph_kernel_nodes != expected_nodes:
            raise RuntimeError(
                f"Expected {expected_nodes} captured kernel nodes, got {graph_kernel_nodes}"
            )
        if args.graph_debug:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if driver.cuGraphDebugDotPrint(
                handle, str(args.output.with_suffix(".dot")).encode(), 0
            ):
                raise RuntimeError("Cannot save control graph debug artifact")
        graph.instantiate()
        graph.replay()
        torch.cuda.synchronize()
        collector.read(expected_count=expected_nodes)
        collector.clear()
        # Warm the same device-side sequence we measure. Individual Python
        # launches can leave the GPU mostly idle despite a long wall-time warmup.
        graph_deadline = time.monotonic() + args.warmup_seconds
        graph_warmup_replays = 0
        while time.monotonic() < graph_deadline:
            graph.replay()
            torch.cuda.synchronize()
            collector.read(expected_count=expected_nodes)
            collector.clear()
            graph_warmup_replays += 1
        sample = snapshot(0)
        assert_available(sample, own_pid=os.getpid(), allowed_graphics=graphics)
        telemetry.append(sample)

        def measured_graph():
            result = []
            for _ in range(launches // pairs_per_replay):
                collector.clear()
                graph.replay()
                torch.cuda.synchronize()
                batch = collector.read(expected_count=expected_nodes)
                # Validate before clearing: surplus or cross-replay records
                # must never disappear when collecting the next chunk.
                result.extend(
                    validate_timestamps(
                        batch, expected_names=names * pairs_per_replay, device=0
                    )
                )
            return result

        if args.monitored:
            raw, monitoring = monitor_call(measured_graph, allowed_graphics=graphics)
        else:
            raw = measured_graph()
        try:
            records = validate_timestamps(
                raw, expected_names=names * launches, device=0
            )
        except ValueError:
            _write(
                args.output,
                dict(
                    role="control",
                    case=case,
                    eligible_for_fit=False,
                    failed="invalid_graph_records",
                    records=raw,
                    expected_names=names * launches,
                ),
            )
            raise
        samples = group_kernel_intervals(
            records, kernels_per_sample=args.kernels_per_sample
        )
    if not args.graph_samples:

        def measured_direct():
            raw = []
            for _ in range(11 * args.kernels_per_sample):
                collector.clear()
                evict()  # Outside the control's timestamp interval.
                launch()
                torch.cuda.synchronize()
                raw.extend(
                    validate_timestamps(
                        collector.read(expected_count=2), expected_names=names, device=0
                    )
                )
            return raw

        if args.monitored:
            raw, monitoring = monitor_call(measured_direct, allowed_graphics=graphics)
        else:
            raw = measured_direct()
        records = validate_timestamps(
            raw, expected_names=names * 11 * args.kernels_per_sample, device=0
        )
        samples = group_kernel_intervals(
            records, kernels_per_sample=args.kernels_per_sample
        )
    check_output(case, out)
    telemetry.append(snapshot(0))
    assert_available(telemetry[-1], own_pid=os.getpid(), allowed_graphics=graphics)
    timings = [sample["latency_us"] for sample in samples]
    median = statistics.median(timings)
    relative_span = (max(timings) - min(timings)) / median
    _write(
        args.output,
        dict(
            schema="triton-viz.gpu-cupti-probe.v1",
            role="control",
            case=case,
            source=source,
            samples=samples,
            kernels_per_sample=args.kernels_per_sample,
            delivery_timeout_seconds=args.delivery_timeout_seconds,
            timestamp_method=args.timestamp_method,
            telemetry=telemetry,
            monitoring=monitoring,
            contaminated=monitoring["contaminated"] if monitoring else None,
            median_us=median,
            relative_span=relative_span,
            unstable=relative_span > 0.15,
            warmup_seconds=args.warmup_seconds,
            warmup_launches=warmup_launches,
            launch_mode="chunked_graph_eviction_control_pairs"
            if args.graph_pairs_per_replay is not None
            else "graph_eviction_control_pairs"
            if args.graph_samples
            else "individual_launches",
            graph_warmup_replays=graph_warmup_replays if args.graph_samples else 0,
            graph_kernel_nodes=graph_kernel_nodes if args.graph_samples else None,
            graph_pairs_per_replay=args.graph_pairs_per_replay,
            metric="cupti_software_serial_group_mean_kernel_us_eviction_unvalidated"
            if args.timestamp_method == "software_serial"
            else "cupti_hes_kernel_us_eviction_unvalidated"
            if args.kernels_per_sample == 1
            else "cupti_hes_group_mean_kernel_us_eviction_unvalidated",
            l2_capacity_bytes=l2_bytes.value,
            eviction_bytes=sweep_bytes,
            eviction_mode=args.eviction_mode,
            cache_counter_validation="pending",
            eligible_for_fit=False,
            library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
            numerical_validation="passed",
            dropped_records=0,
        ),
    )
    print(
        f"Saved {len(samples)} {args.timestamp_method} samples; eviction validation remains pending"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
