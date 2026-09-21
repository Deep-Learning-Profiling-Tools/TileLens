"""Audit the entire independent local-footprint grid without fitting latency."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from triton_viz.tools.gpu_cost_model_pipeline import _write
from triton_viz.tools.gpu_cupti_perturbation_audit import audit_log
from triton_viz.tools.gpu_local_cache_counter_collect import (
    METRICS,
    footprint_grid,
    parse_counters,
)


def audit_grid(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest.get("role") != "control"
        or manifest.get("eligible_for_fit") is not False
        or manifest.get("cases") != footprint_grid()
        or manifest.get("counter_only") is not True
        or manifest.get("iterations") != 65536
        or manifest.get("workload") != "local"
        or manifest.get("metrics") != list(METRICS)
    ):
        raise ValueError("Require the complete declared control footprint grid")
    rows = []
    for case in footprint_grid():
        p, s = case["programs"], case["local_slots"]
        stem = f"local_s{s}_p{p}_i65536"
        row = json.loads((root / (stem + ".json")).read_text())
        if (
            row.get("role") != "control"
            or row.get("eligible_for_fit") is not False
            or row.get("status") != "complete"
            or any(row.get(k) != v for k, v in case.items())
        ):
            raise ValueError("Incomplete or relabeled control")
        raw, counters = (
            (root / (stem + ".log")).read_bytes(),
            (root / (stem + ".csv")).read_bytes(),
        )
        if (
            hashlib.sha256(raw).hexdigest() != row["log_sha256"]
            or hashlib.sha256(counters).hexdigest() != row["counter_sha256"]
        ):
            raise ValueError("Changed raw evidence")
        native = audit_log(
            raw.decode(),
            mode="none",
            programs=p,
            iterations=65536,
            workload="local",
            local_slots=s,
            counter_only=True,
        )
        parsed = parse_counters(counters.decode())
        if native != row["native_audit"] or parsed != row["counters"]:
            raise ValueError("Stored audit differs from raw evidence")
        lines = counters.decode().splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith('"ID",'))
        for metric in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
            if (
                metric["Grid Size"] != f"({p}, 1, 1)"
                or metric["Block Size"] != "(128, 1, 1)"
                or metric["Device"] != "0"
            ):
                raise ValueError("Counter launch geometry mismatch")
        monitor = row["monitoring"]
        if (
            monitor["contaminated"]
            or monitor["rejection_reasons"]
            or monitor["returncode"] != 0
            or monitor.get("own_process_group") is not True
            or len(monitor["samples"]) < 2
        ):
            raise ValueError("Rejected monitoring")
        for sample in monitor["samples"]:
            groups = sample.get("observed_process_groups", {})
            if (
                any(
                    sample[k] != manifest["baseline"][k]
                    for k in ("uuid", "driver", "index")
                )
                or any(
                    proc["pid"] != monitor["child_pid"]
                    and groups.get(str(proc["pid"])) != monitor["child_pid"]
                    for proc in sample["processes"]
                )
                or any(
                    proc["pid"] not in manifest["allowed_graphics"]
                    for proc in sample["graphics_processes"]
                )
            ):
                raise ValueError("Foreign process or changed hardware identity")
        # 128 FP32 threads request sixteen 32-byte sectors per slot. This is
        # source traffic, not inferred DRAM bytes or inferred compiler spilling.
        expected = p * 16 * (65536 + s)
        metrics = parsed["metrics"]
        rows.append(
            dict(
                **case,
                footprint_bytes=p * 128 * 4 * s,
                source_local_load_sectors=expected,
                source_local_store_sectors=expected,
                local_load_exact=metrics[METRICS[0]] == expected,
                local_store_exact=metrics[METRICS[1]] == expected,
                **parsed,
            )
        )
    return dict(
        role="control",
        eligible_for_fit=False,
        count=len(rows),
        rows=rows,
        caveat="All controls retained. Counter diagnosis only; no latency CV or source spill-allocation validation.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Require a fresh audit output")
    result = audit_grid(args.root)
    _write(args.output, result)
    print(f"Audited {result['count']} controls; no latency admission")


if __name__ == "__main__":
    main()
