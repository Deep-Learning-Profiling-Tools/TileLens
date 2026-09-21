"""Profile the three declared native local-memory controls; no latency fitting."""

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


def parse_counters(text):
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
            or name not in METRICS
            or name in values
        ):
            raise ValueError("Unexpected launch, kernel or metric")
        value = float(row["Metric Value"].replace(",", ""))
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid counter value")
        if row["Metric Unit"] != ("%" if name.endswith(".pct") else "sector"):
            raise ValueError("Unexpected counter unit")
        values[name] = value
    if set(values) != set(METRICS) or values[METRICS[2]] > 100:
        raise ValueError("Incomplete or invalid counter set")
    reads, hits, misses = (values[k] for k in METRICS[3:])
    if reads <= 0 or hits + misses <= 0:
        raise ValueError("Empty L2 measurement")
    return dict(
        metrics=values,
        l2_hit_fraction=hits / (hits + misses),
        replay_count_disagreement=abs(reads - hits - misses) / reads,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncu", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-idle-graphics", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Require fresh control counter output")
    ncu, binary = args.ncu.resolve(strict=True), args.binary.resolve(strict=True)
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
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    _write(
        args.output / "manifest.json",
        dict(
            role="control",
            eligible_for_fit=False,
            programs=[48, 96, 384],
            iterations=65536,
            workload="local",
            metrics=list(METRICS),
            baseline=baseline,
            binary_sha256=digest,
            ncu_version=version,
            protocol="first_body_after_eviction_kernel_replay_all_cache_no_clock_change",
        ),
    )
    for programs in (48, 96, 384):
        stem = f"local_p{programs}_i65536"
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
            ",".join(METRICS),
            "--csv",
            "--log-file",
            str(counters),
            str(binary),
            "none",
            "unused",
            str(programs),
            "65536",
        ]
        result = dict(
            role="control", eligible_for_fit=False, programs=programs, command=command
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
            )
            result["counter_sha256"] = hashlib.sha256(counters.read_bytes()).hexdigest()
            result["counters"] = parse_counters(counters.read_text())
            result["status"] = "complete"
        except Exception as error:
            result.update(status="failed", error=str(error))
            _write(args.output / (stem + ".json"), result)
            raise
        _write(args.output / (stem + ".json"), result)
        print(stem, "complete counters_only", flush=True)


if __name__ == "__main__":
    main()
