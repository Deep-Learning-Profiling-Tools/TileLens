"""Verify monitored control HES batches; do not fit or relabel old protocols."""

import argparse
import json
import statistics
from pathlib import Path

from microbench.gpu.harness.cupti import validate_timestamps
from triton_viz.tools.gpu_cost_model_pipeline import _write


def validate_batch(row, case, *, library_sha256):
    if row.get("role") != "control" or row["case"] != case:
        raise ValueError("Control identity mismatch")
    repetitions = row.get("kernels_per_sample", 1)
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 1
    ):
        raise ValueError("Invalid replication count")
    monitor = row.get("monitoring")
    method = row.get("timestamp_method", "hes")
    launch = row.get("launch_mode")
    if method not in {"hes", "software_serial"} or launch not in {
        "graph_eviction_control_pairs",
        "individual_launches",
    }:
        raise ValueError("Unverified HES provenance or unknown CUPTI protocol")
    if (
        row.get("contaminated") is not False
        or not monitor
        or monitor.get("contaminated") is not False
        or monitor.get("rejection_reasons")
        or len(monitor.get("samples", [])) < 2
    ):
        raise ValueError("Missing or rejected process monitoring")
    if (
        row.get("dropped_records") != 0
        or row.get("eviction_mode", "torch_zero") != "torch_zero"
        or row.get("numerical_validation") != "passed"
        or row.get("graph_kernel_nodes")
        != (22 * repetitions if launch == "graph_eviction_control_pairs" else None)
        or row.get("library_sha256") != library_sha256
    ):
        raise ValueError("Unverified HES provenance")
    if (
        row.get("metric")
        != (
            "cupti_software_serial_group_mean_kernel_us_eviction_unvalidated"
            if method == "software_serial"
            else "cupti_hes_kernel_us_eviction_unvalidated"
            if repetitions == 1
            else "cupti_hes_group_mean_kernel_us_eviction_unvalidated"
        )
        or row.get("eligible_for_fit") is not False
    ):
        raise ValueError("Unexpected or silently relabeled timing protocol")
    if (
        row["l2_capacity_bytes"] <= 0
        or row["eviction_bytes"] != 2 * row["l2_capacity_bytes"]
    ):
        raise ValueError("Unexpected L2 sweep size")
    samples = row["samples"]
    if len(samples) != 11 or any(len(s["records"]) != 2 * repetitions for s in samples):
        raise ValueError("Require all 11 sweep/control pairs")
    names = [r["name"] for r in samples[0]["records"][:2]]
    if names[0] == names[1]:
        raise ValueError("Sweep and control must be distinguishable")
    raw = [r for s in samples for r in s["records"]]
    ordered = validate_timestamps(
        raw, expected_names=names * 11 * repetitions, device=0
    )
    if ordered != raw:
        raise ValueError("Stored sample pairs are not in execution order")
    individual = [
        [(r["end_ns"] - r["start_ns"]) / 1000 for r in s["records"][1::2]]
        for s in samples
    ]
    if repetitions > 1 and individual != [
        s.get("kernel_latencies_us") for s in samples
    ]:
        raise ValueError("All replicated raw intervals must be retained")
    timings = [statistics.mean(values) for values in individual]
    if timings != [s["latency_us"] for s in samples]:
        raise ValueError("Timing differs from kernel-only HES intervals")
    median = statistics.median(timings)
    span = (max(timings) - min(timings)) / median
    if (
        row["median_us"] != median
        or row["relative_span"] != span
        or row["unstable"]
        or span > 0.15
    ):
        raise ValueError("Invalid or unstable timing summary")
    return dict(latency_us=median, relative_span=span, sample_count=11)


def audit(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("role") != "control" or manifest.get("monitored") is not True:
        raise ValueError("Require a declared monitored control collection")
    if not manifest.get("library_sha256"):
        raise ValueError("Missing collector library fingerprint")
    cases = manifest["cases"]
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate declared controls")
    rows = []
    for case in cases:
        path = root / "controls" / (case["id"] + ".json")
        if not path.exists():
            rows.append(dict(case=case, status="missing"))
            continue
        row = json.loads(path.read_text())
        if row.get("kernels_per_sample", 1) != manifest.get("kernels_per_sample", 1):
            raise ValueError("Mixed sample replication protocols")
        if row.get("timestamp_method", "hes") != manifest.get(
            "timestamp_method", "hes"
        ) or row.get("launch_mode") != {
            "graph": "graph_eviction_control_pairs",
            "individual": "individual_launches",
        }.get(manifest.get("launch_mode", "graph")):
            raise ValueError("Mixed CUPTI timestamp or launch protocols")
        summary = validate_batch(row, case, library_sha256=manifest["library_sha256"])
        attempt = row["accepted_attempt"]
        if (
            isinstance(attempt, bool)
            or not isinstance(attempt, int)
            or not 1 <= attempt <= 3
        ):
            raise ValueError("Invalid bounded attempt index")
        original = json.loads(
            (root / "attempts" / case["id"] / f"{attempt}.json").read_text()
        )
        if {
            k: v
            for k, v in row.items()
            if k not in {"accepted_attempt", "attempt_source"}
        } != original:
            raise ValueError("Accepted batch differs from archived attempt")
        rows.append(
            dict(case=case, status="complete", accepted_attempt=attempt, **summary)
        )
    return dict(
        role="control",
        count=len(rows),
        rows=rows,
        measurement_integrity_passed=bool(rows)
        and all(r["status"] == "complete" for r in rows),
        eligible_for_fit=False,
        caveat="Timing integrity only. Cache validation, protocol identity, source features and CV gates remain separate requirements.",
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
    print(
        {
            k: result[k]
            for k in ("count", "measurement_integrity_passed", "eligible_for_fit")
        }
    )


if __name__ == "__main__":
    main()
