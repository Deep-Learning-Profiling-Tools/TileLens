"""Audit declared dot-control cache counters, including zero local requests.

Local requests and cache lookups can be replayed separately. Preserve their
count disagreement rather than equating aggregate hit rate with load hits.
No measured counters enter the source-only prediction API or a latency fit.
"""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

from triton_viz.tools.gpu_control_resources import selected_controls
from triton_viz.tools.gpu_cost_model_pipeline import _write
from triton_viz.tools.gpu_local_counter_audit import validate_monitor
from triton_viz.tools.gpu_local_counter_collect import CACHE_METRICS, ISSUE_METRICS


def parse_issue(text, *, retain_invalid_stalls=False):
    lines = text.splitlines()
    header = next((i for i, line in enumerate(lines) if line.startswith('"ID",')), None)
    if header is None:
        raise ValueError("Missing instruction counter header")
    values = {}
    for row in csv.DictReader(io.StringIO("\n".join(lines[header:]))):
        name = row["Metric Name"]
        percent = name.endswith(".pct")
        if (
            row["ID"] != "0"
            or row["Kernel Name"] != "geometry_dot"
            or name not in ISSUE_METRICS
            or name in values
            or row["Metric Unit"] != ("%" if percent else "inst")
        ):
            raise ValueError("Unexpected instruction counter launch, metric or unit")
        value = float(row["Metric Value"].replace(",", ""))
        if (
            not math.isfinite(value)
            or value < 0
            or (percent and value > 100 and not retain_invalid_stalls)
            or (not percent and not value.is_integer())
        ):
            raise ValueError("Invalid instruction counter value")
        values[name] = value if percent else int(value)
    if set(values) != set(ISSUE_METRICS):
        raise ValueError("Incomplete instruction counters")
    return values


def parse(text):
    lines = text.splitlines()
    header = next((i for i, line in enumerate(lines) if line.startswith('"ID",')), None)
    if header is None:
        raise ValueError("Missing cache counter header")
    values = {}
    for row in csv.DictReader(io.StringIO("\n".join(lines[header:]))):
        name = row["Metric Name"]
        if (
            row["ID"] != "0"
            or row["Kernel Name"] != "geometry_dot"
            or name not in CACHE_METRICS
            or name in values
            or row["Metric Unit"] != "sector"
        ):
            raise ValueError("Unexpected cache counter launch, metric or unit")
        raw = row["Metric Value"].replace(",", "")
        if not raw.isdecimal():
            raise ValueError("Require nonnegative integer sector counts")
        values[name] = int(raw)
    if set(values) != set(CACHE_METRICS):
        raise ValueError("Incomplete cache counters")
    result = {}
    for op, indices in (("LDL", (0, 2, 3)), ("STL", (1, 4, 5)), ("L2_read", (6, 7, 8))):
        requests, hits, misses = (values[CACHE_METRICS[i]] for i in indices)
        lookups = hits + misses
        result[op] = dict(
            requests=requests,
            hits=hits,
            misses=misses,
            hit_fraction=hits / lookups if lookups else None,
            replay_count_difference=lookups - requests,
        )
    return result


def audit(root, *, issue_work=False, retain_invalid_stalls=False):
    if retain_invalid_stalls and not issue_work:
        raise ValueError("Invalid-stall retention only applies to issue diagnostics")
    manifest = json.loads((root / "manifest.json").read_text())
    suite = manifest.get("suite")
    if suite not in {"pressure", "pressure_pipeline", "resource_dot"}:
        raise ValueError("Require declared dot controls")
    cases = selected_controls(suite)
    if (
        manifest.get("role") != "control"
        or manifest.get("cases") != cases
        or manifest.get("metrics")
        != list(ISSUE_METRICS if issue_work else CACHE_METRICS)
        or manifest.get("issue_work_phase" if issue_work else "cache_lookup_phase")
        is not True
        or manifest.get("monitored") is not True
    ):
        raise ValueError("Cache protocol or control declaration mismatch")
    rows = []
    for case in cases:
        path = root / (case["id"] + ".csv")
        validate_monitor(path, manifest)
        if (
            f"control={case['id']} numerical=passed profiler_timing_not_for_fit"
            not in path.with_suffix(".log").read_text()
        ):
            raise ValueError("Missing numerical validation")
        counters = (
            parse_issue(path.read_text(), retain_invalid_stalls=retain_invalid_stalls)
            if issue_work
            else parse(path.read_text())
        )
        errors = (
            {k: v for k, v in counters.items() if k.endswith(".pct") and v > 100}
            if issue_work
            else {}
        )
        rows.append(
            dict(
                case=case,
                csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                counters=counters,
                **(dict(invalid_stall_percentages=errors) if issue_work else {}),
            )
        )
    return dict(
        role="control",
        count=len(rows),
        rows=rows,
        complete=True,
        **(
            dict(
                stall_measurement_integrity_passed=not any(
                    r["invalid_stall_percentages"] for r in rows
                )
            )
            if issue_work
            else {}
        ),
        eligible_for_fit=False,
        released_model=None,
        caveat="Control instruction/stall diagnostics only. Warp-active stall fractions are not additive latency components or source prediction inputs. No profiler timing fit."
        if issue_work
        else "All declared controls retained. L2 reads include global and local traffic, not local-only misses. Replay count disagreement remains explicit; no latency fitting.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--issue-work", action="store_true")
    parser.add_argument("--retain-invalid-stalls", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Fresh audit output required")
    report = audit(
        args.root,
        issue_work=args.issue_work,
        retain_invalid_stalls=args.retain_invalid_stalls,
    )
    _write(args.output, report)
    print(dict(count=report["count"], complete=report["complete"]))


if __name__ == "__main__":
    main()
