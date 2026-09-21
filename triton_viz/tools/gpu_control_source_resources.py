"""Observe declared resource controls on CPU; never read compiler artifacts."""

import argparse
from collections import Counter
from pathlib import Path

from microbench.gpu.common.cases import load_cases
from triton_viz.performance.calibration import stable_digest
from triton_viz.performance.gpu_resources import (
    source_liveness_features,
    source_resource_features,
)
from triton_viz.tools.gpu_cost_model_pipeline import _write


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", choices=("pressure", "resource_transfer"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-loops", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Use a fresh source observation root")
    cases = load_cases(args.suite, "control")
    _write(args.output / "manifest.json", dict(role="control", cases=cases))
    import torch
    import triton
    from microbench.gpu.tests.coverage.kernels import prepare, check_output
    from triton_viz.performance.triton_observe import observe

    def forbidden(*args, **kwargs):
        raise RuntimeError("Resource source observation forbids compilation and CUDA")

    old_compile, old_init = triton.compile, torch.cuda._lazy_init
    old_threads = torch.get_num_threads()
    try:
        triton.compile = torch.cuda._lazy_init = forbidden
        torch.set_num_threads(1)
        for case in cases:
            kernel, grid, inputs, out = prepare(case, "cpu")
            source = observe(
                kernel,
                grid,
                *inputs,
                num_warps=case["num_warps"],
                num_stages=case["num_stages"],
                capture_loops=args.capture_loops,
            )
            check_output(case, out)
            features, precision, reasons = source_resource_features(source)
            liveness = source_liveness_features(source)
            shapes = sorted(
                {
                    tuple(tuple(s) for s in e["input_shapes"][:2])
                    for e in source["events"]
                    if e["op"] == "dot"
                }
            )
            _write(
                args.output / "controls" / (case["id"] + ".json"),
                dict(
                    role="control",
                    case=case,
                    source_features=features,
                    source_precision=precision,
                    ood_reasons=reasons,
                    source_liveness=liveness,
                    dot_shapes=shapes,
                    operation_counts=dict(Counter(e["op"] for e in source["events"])),
                    source_digest=stable_digest(source),
                    program_count=source["program_count"],
                    numerical_validation="passed",
                    compile_and_cuda_forbidden=True,
                    **(
                        {
                            "loop_trace": source["loop_trace"],
                            "parent_source_digest": stable_digest(
                                {k: v for k, v in source.items() if k != "loop_trace"}
                            ),
                        }
                        if args.capture_loops
                        else {}
                    ),
                ),
            )
            print(case["id"], liveness, flush=True)
    finally:
        triton.compile, torch.cuda._lazy_init = old_compile, old_init
        torch.set_num_threads(old_threads)


if __name__ == "__main__":
    main()
