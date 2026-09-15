"""CPU-only correctness, equal-work and dependency audit of paired controls."""

import argparse
from collections import Counter
import json

from microbench.gpu.common.cases import paired_controls
from microbench.gpu.tests.coverage.kernels import prepare, check_output
from triton_viz.performance.gpu import expand
from triton_viz.performance.gpu_distributions import (
    distribution_features,
    PROGRAM_FEATURES,
)


def audit_pair(cases):
    import triton_viz
    from triton_viz.performance.triton_observe import observe

    if len(cases) != 2 or {c["topology"] for c in cases} != {"serial", "parallel"}:
        raise ValueError("Expected a serial/parallel pair")
    if (
        len({c["pair_id"] for c in cases}) != 1
        or len({c["cv_group"] for c in cases}) != 1
    ):
        raise ValueError("Pair identity/CV group mismatch")
    records = {}
    for case in cases:
        try:
            kernel, grid, args, output = prepare(case, "cpu")
            source = observe(
                kernel,
                grid,
                *args,
                num_warps=case["num_warps"],
                num_stages=case["num_stages"],
            )
            check_output(case, output)
            work = expand(source, sm_count=48)
            if source["program_count"] != 1 or work["ood_reasons"]:
                raise ValueError("Expected a supported single-program trace")
            features, _ = distribution_features(source)
            events = source["events"]
            stores = [e for e in events if e["op"] in {"store", "raw_store"}]
            if len(stores) != 2 * case["repeat"]:
                raise ValueError("Missing stage stores")
            live = set()
            pending = [e["seq"] for e in stores]
            while pending:
                seq = pending.pop()
                if seq not in live:
                    live.add(seq)
                    pending.extend(events[seq]["dependencies"])
            floating = [
                e
                for e in events
                if e["op"] in {"binary_op", "unary_op", "reduce_sum", "reduce_max"}
                and e["dtype"] == "fp32"
            ]
            if any(e["seq"] not in live for e in floating):
                raise ValueError("Floating work is disconnected from stored outputs")
            records[case["topology"]] = dict(
                aggregate=work["features"],
                distribution=features,
                floating_work=Counter(
                    (e["op"], e.get("primitive"), tuple(e["shape"])) for e in floating
                ),
                bytes=sum(e.get("bytes", 0) for e in events),
            )
        finally:
            triton_viz.clear()
    serial, parallel = records["serial"], records["parallel"]
    for field in ("aggregate", "floating_work", "bytes"):
        if serial[field] != parallel[field]:
            raise ValueError(f"Unequal source work: {field}")
    for feature in PROGRAM_FEATURES:
        if serial["distribution"][feature] != parallel["distribution"][feature]:
            raise ValueError(f"Unequal per-program work: {feature}")
    component = {"alu": "alu", "sfu": "sfu", "sum": "reduction", "max": "reduction"}[
        cases[0]["operation"]
    ]
    key = f"path_{component}_p90"
    serial_path, parallel_path = (
        record["distribution"][key] for record in (serial, parallel)
    )
    if parallel_path <= 0 or serial_path != 2 * parallel_path:
        raise ValueError(
            f"Expected 2:1 dependency depth: {serial_path}/{parallel_path}"
        )
    return dict(
        pair_id=cases[0]["pair_id"],
        cv_group=cases[0]["cv_group"],
        component=component,
        serial_path=serial_path,
        parallel_path=parallel_path,
        aggregate=serial["aggregate"],
        bytes=serial["bytes"],
        passed=True,
    )


def main():
    cases = paired_controls()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for index in range(0, len(cases), 2):
        print(
            json.dumps(audit_pair(cases[index : index + 2]), sort_keys=True), flush=True
        )


if __name__ == "__main__":
    main()
