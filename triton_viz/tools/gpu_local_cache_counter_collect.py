"""Profile declared native local-memory controls; no latency fitting."""

import argparse
import csv
import hashlib
import io
import math
import subprocess
from pathlib import Path

from microbench.gpu.harness.measure import snapshot, assert_available
from triton_viz.tools.gpu_cost_model_pipeline import _write
from triton_viz.tools.gpu_cupti_perturbation_collect import monitored_process
from triton_viz.tools.gpu_cupti_perturbation_audit import audit_log

METRICS = (
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sectors_op_read.sum",
    "lts__t_sectors_op_read_lookup_hit.sum",
    "lts__t_sectors_op_read_lookup_miss.sum",
)
LOCAL_LOOKUP_METRICS = tuple(
    f"l1tex__t_sectors_pipe_lsu_mem_local_op_{op}_lookup_{outcome}.sum"
    for op in ("ld", "st")
    for outcome in ("hit", "miss")
)


def footprint_grid():
    """Independent working-set / concurrency axes fixed before collection."""
    return [
        dict(local_slots=s, programs=p)
        for s in (32, 64, 128, 256)
        for p in (48, 96, 192, 384)
    ]


def parse_counters(text, *, local_lookups=False):
    expected_metrics = METRICS + LOCAL_LOOKUP_METRICS if local_lookups else METRICS
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith('"ID",')), None)
    if start is None:
        raise ValueError("Missing ncu CSV header")
    values = {}
    for row in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
        name = row["Metric Name"]
        if (
            row["ID"] != "0"
            or "perturbation_body" not in row["Kernel Name"]
            or name not in expected_metrics
            or name in values
        ):
            raise ValueError("Unexpected launch, kernel or metric")
        value = float(row["Metric Value"].replace(",", ""))
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid counter value")
        if row["Metric Unit"] != ("%" if name.endswith(".pct") else "sector"):
            raise ValueError("Unexpected counter unit")
        values[name] = value
    if set(values) != set(expected_metrics) or values[METRICS[2]] > 100:
        raise ValueError("Incomplete or invalid counter set")
    reads, hits, misses = (values[k] for k in METRICS[3:])
    if reads <= 0 or hits + misses <= 0:
        raise ValueError("Empty L2 measurement")
    lookup = {}
    if local_lookups:
        for index, op in enumerate(("load", "store")):
            hit, miss = (
                values[k] for k in LOCAL_LOOKUP_METRICS[2 * index : 2 * index + 2]
            )
            total = values[METRICS[index]]
            if hit + miss <= 0 or total <= 0:
                raise ValueError("Empty local lookup measurement")
            lookup[op] = dict(
                hit_fraction=hit / (hit + miss),
                replay_count_disagreement=abs(total - hit - miss) / total,
            )
    return dict(
        metrics=values,
        **(dict(local_lookups=lookup) if local_lookups else {}),
        l2_hit_fraction=hits / (hits + misses),
        replay_count_disagreement=abs(reads - hits - misses) / reads,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncu", type=Path, required=True)
    binaries = parser.add_mutually_exclusive_group(required=True)
    binaries.add_argument("--binary", type=Path)
    binaries.add_argument(
        "--footprint-binaries",
        type=Path,
        nargs=4,
        help="Precompiled local-slot binaries in order: 32 64 128 256",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-idle-graphics", action="store_true")
    parser.add_argument("--local-lookups", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Require fresh control counter output")
    ncu = args.ncu.resolve(strict=True)
    grid = args.footprint_binaries is not None
    binary_map = (
        dict(zip((32, 64, 128, 256), args.footprint_binaries))
        if grid
        else {128: args.binary}
    )
    binary_map = {s: p.resolve(strict=True) for s, p in binary_map.items()}
    digests = {
        s: hashlib.sha256(p.read_bytes()).hexdigest() for s, p in binary_map.items()
    }
    cases = (
        footprint_grid()
        if grid
        else [dict(local_slots=128, programs=p) for p in (48, 96, 384)]
    )
    metrics = METRICS + LOCAL_LOOKUP_METRICS if args.local_lookups else METRICS
    baseline = snapshot(0)
    graphics = (
        tuple(p["pid"] for p in baseline["graphics_processes"])
        if args.allow_idle_graphics
        else ()
    )
    assert_available(baseline, allowed_graphics=graphics)
    version = subprocess.run(
        [str(ncu), "--version"], check=True, capture_output=True, text=True, timeout=15
    ).stdout
    _write(
        args.output / "manifest.json",
        dict(
            role="control",
            eligible_for_fit=False,
            programs=[48, 96, 192, 384] if grid else [48, 96, 384],
            cases=cases,
            counter_only=grid,
            allowed_graphics=list(graphics),
            iterations=65536,
            workload="local",
            metrics=list(metrics),
            baseline=baseline,
            binary_sha256=digests,
            binaries={s: str(p) for s, p in binary_map.items()},
            ncu_version=version,
            protocol="first_body_after_eviction_kernel_replay_all_cache_no_clock_change",
        ),
    )
    for case in cases:
        programs, slots = case["programs"], case["local_slots"]
        binary, digest = binary_map[slots], digests[slots]
        stem = (
            f"local_s{slots}_p{programs}_i65536"
            if grid
            else f"local_p{programs}_i65536"
        )
        log, counters = args.output / (stem + ".log"), args.output / (stem + ".csv")
        command = [
            str(ncu),
            "--replay-mode",
            "kernel",
            "--cache-control",
            "all",
            "--clock-control",
            "none",
            "--kernel-name",
            "regex:perturbation_body",
            "--launch-count",
            "1",
            "--metrics",
            ",".join(metrics),
            "--csv",
            "--log-file",
            str(counters),
            str(binary),
            "none",
            "unused",
            str(programs),
            "65536",
        ]
        if grid:
            command.append("counter_only")
        result = dict(
            role="control",
            eligible_for_fit=False,
            programs=programs,
            local_slots=slots,
            command=command,
        )
        try:
            if hashlib.sha256(binary.read_bytes()).hexdigest() != digest:
                raise ValueError("Native binary changed")
            result["monitoring"] = monitored_process(
                command,
                log,
                allowed_graphics=graphics,
                timeout=240,
                own_process_group=True,
            )
            result["log_sha256"] = hashlib.sha256(log.read_bytes()).hexdigest()
            if (
                result["monitoring"]["contaminated"]
                or result["monitoring"]["returncode"]
            ):
                raise ValueError("Rejected profiler process or monitoring")
            # Validate the entire native workload including final exact numeric
            # check. Profiler-perturbed timestamps are retained, never fitted.
            result["native_audit"] = audit_log(
                log.read_text(),
                mode="none",
                programs=programs,
                iterations=65536,
                workload="local",
                local_slots=slots,
                counter_only=grid,
            )
            result["counter_sha256"] = hashlib.sha256(counters.read_bytes()).hexdigest()
            result["counters"] = parse_counters(
                counters.read_text(), local_lookups=args.local_lookups
            )
            result["status"] = "complete"
        except Exception as error:
            result.update(status="failed", error=str(error))
            _write(args.output / (stem + ".json"), result)
            raise
        _write(args.output / (stem + ".json"), result)
        print(stem, "complete counters_only", flush=True)


if __name__ == "__main__":
    main()
