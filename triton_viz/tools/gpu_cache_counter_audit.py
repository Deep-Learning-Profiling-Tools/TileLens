"""Audit all declared cache-counter controls; never fit latency or read targets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
from pathlib import Path

from triton_viz.tools.gpu_cost_model_pipeline import _write
from microbench.gpu.common.cache_controls import cache_declaration

METRICS = {
    "lts__t_sectors_op_read.sum": "reads",
    "lts__t_sectors_op_read_lookup_hit.sum": "hits",
    "lts__t_sectors_op_read_lookup_miss.sum": "misses",
}


def parse_counters(text, *, range_mode=False):
    lines = text.splitlines()
    header = next((i for i, line in enumerate(lines) if line.startswith('"ID",')), None)
    if header is None:
        raise ValueError("No complete ncu CSV table")
    launches = {}
    for row in csv.DictReader(io.StringIO("\n".join(lines[header:]))):
        if (
            row["Kernel Name"] != ("range" if range_mode else "cache_read_control")
            or row["Metric Name"] not in METRICS
        ):
            raise ValueError("Unexpected kernel or metric in control report")
        index = int(row["ID"])
        metric = METRICS[row["Metric Name"]]
        sample = launches.setdefault(index, {})
        if metric in sample:
            raise ValueError("Duplicate control metric")
        value = float(row["Metric Value"].replace(",", ""))
        if not value >= 0 or value == float("inf"):
            raise ValueError("Invalid counter value")
        sample[metric] = value
    expected_ids = {0} if range_mode else {0, 1, 2}
    if set(launches) != expected_ids or any(
        set(row) != set(METRICS.values()) for row in launches.values()
    ):
        raise ValueError("Require all declared launches/ranges and all counter metrics")
    for sample in launches.values():
        if sample["reads"] <= 0 or sample["hits"] + sample["misses"] <= 0:
            raise ValueError("Empty counter measurement")
        # Metrics are collected over different application replay passes. Keep
        # their disagreement visible, not silently enforce sum equality.
        sample["replay_count_disagreement"] = (
            abs(sample["hits"] + sample["misses"] - sample["reads"]) / sample["reads"]
        )
        sample["hit_fraction"] = sample["hits"] / (sample["hits"] + sample["misses"])
    return [launches[index] for index in sorted(expected_ids)]


def audit_ranges(root, matrix="legacy"):
    declaration = cache_declaration(matrix)
    rows = []
    for mib in declaration["working_set_mib"]:
        for eviction in declaration["evictions"]:
            path = root / f"{mib}_{eviction}.csv"
            log = path.with_suffix(".log")
            record = dict(
                working_set_mib=mib,
                eviction=eviction,
                path=str(path),
                cv_group=f"cache_capacity_mib{mib}",
            )
            try:
                text = path.read_text()
                counters = parse_counters(text, range_mode=True)[0]
                if "numerical=passed" not in log.read_text():
                    raise ValueError("Missing numerical validation")
                record.update(
                    status="complete",
                    counters=counters,
                    sha256=hashlib.sha256(text.encode()).hexdigest(),
                )
            except (OSError, ValueError) as exc:
                record.update(status="incomplete", error=str(exc))
            rows.append(record)
    complete = all(row["status"] == "complete" for row in rows)
    reliable = complete and all(
        row["counters"]["replay_count_disagreement"] <= 0.05 for row in rows
    )
    delta = None
    if reliable:
        # With 3 MiB input + 3 MiB output fitting in 24 MiB L2, a successful
        # sweep before the third read adds one input footprint of misses.
        baseline = rows[0]["counters"]
        swept = rows[1]["counters"]
        delta = (swept["misses"] - baseline["misses"]) / (3 * 1024 * 1024 / 32)
    return dict(
        schema="triton-viz.gpu-cache-range-audit.v1",
        role="control",
        matrix=matrix,
        declaration=declaration,
        rows=rows,
        complete=complete,
        replay_counts_consistent=bool(reliable),
        zero_sweep_added_miss_footprints=delta,
        protocol="range replay; tool cold-starts range; three reads with optional sweep before third; counters aggregate all launches including sweep",
        caveat="Read sweep adds its own traffic; do not interpret range-wide hit fraction as the third kernel hit rate.",
        eligible_for_latency_fit=False,
    )


def audit(root):
    rows = []
    for mib in (3, 12, 48):
        for eviction in ("none", "zero", "read"):
            # Keep unsuccessful attempts; choose the first complete attempt,
            # never whichever attempt has the most desirable hit rate.
            attempts = []
            accepted = None
            for path in sorted(root.glob(f"matrix_{mib}_{eviction}*.csv")):
                raw = path.read_text()
                try:
                    launches = parse_counters(raw)
                    log_path = path.with_suffix(".log")
                    if (
                        not log_path.exists()
                        or "numerical=passed" not in log_path.read_text()
                    ):
                        raise ValueError(
                            "Missing successful control numerical validation"
                        )
                    error = None
                except ValueError as exc:
                    launches, error = None, str(exc)
                attempts.append(
                    dict(
                        path=str(path),
                        sha256=hashlib.sha256(raw.encode()).hexdigest(),
                        error=error,
                    )
                )
                if accepted is None and launches is not None:
                    accepted = dict(path=str(path), launches=launches)
            rows.append(
                dict(
                    working_set_mib=mib,
                    eviction=eviction,
                    attempts=attempts,
                    accepted=accepted,
                )
            )
    complete = all(row["accepted"] is not None for row in rows)
    # These are counter sanity criteria, not fitted latency coefficients.
    reliable = complete and all(
        launch["replay_count_disagreement"] <= 0.05
        for row in rows
        for launch in row["accepted"]["launches"]
    )
    warm = next(
        row for row in rows if row["working_set_mib"] == 3 and row["eviction"] == "none"
    )
    positive_control = (
        warm["accepted"] is not None
        and all(
            launch["replay_count_disagreement"] <= 0.05
            for launch in warm["accepted"]["launches"]
        )
        and warm["accepted"]["launches"][0]["hit_fraction"] < 0.05
        and all(
            launch["hit_fraction"] > 0.95 for launch in warm["accepted"]["launches"][1:]
        )
    )
    validated = {
        eviction: bool(
            reliable
            and positive_control
            and all(
                row["accepted"]["launches"][2]["hit_fraction"] < 0.05
                for row in rows
                if row["eviction"] == eviction
            )
        )
        for eviction in ("zero", "read")
    }
    return dict(
        schema="triton-viz.gpu-cache-counter-audit.v1",
        role="control",
        rows=rows,
        complete=complete,
        replay_counts_consistent=bool(reliable),
        cold_to_warm_positive_control=bool(positive_control),
        validated_eviction=validated,
        scope="3/12/48 MiB contiguous .cg reads, two warm launches, 2x hardware L2 sweep; not arbitrary allocation/cache policy coverage",
        eligible_for_latency_fit=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--range-mode", action="store_true")
    parser.add_argument("--matrix", choices=("legacy", "capacity"), default="legacy")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Use a fresh audit path")
    if args.matrix != "legacy" and not args.range_mode:
        raise ValueError("Capacity controls require the declared range protocol")
    report = (
        audit_ranges(args.root, args.matrix) if args.range_mode else audit(args.root)
    )
    _write(args.output, report)
    print(
        {
            key: report[key]
            for key in (
                "complete",
                "cold_to_warm_positive_control",
                "validated_eviction",
            )
            if key in report
        }
    )


if __name__ == "__main__":
    main()
