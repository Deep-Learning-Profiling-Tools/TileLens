"""Control-only interval overlap and endpoint SM-placement diagnostics."""

import argparse
import json
from collections import Counter
from pathlib import Path

from triton_viz.tools.gpu_cost_model_pipeline import _write
from triton_viz.tools.gpu_local_cache_counter_audit import audit_grid
from triton_viz.tools.gpu_local_cache_counter_collect import footprint_grid


def interval_overlap(intervals):
    """Half-open interval overlap, not an occupancy or issue-rate measurement."""
    if not intervals or any(
        not isinstance(a, int) or not isinstance(b, int) or a <= 0 or b <= a
        for a, b in intervals
    ):
        raise ValueError("Require positive ordered integer intervals")
    first = min(a for a, _ in intervals)
    last = max(b for _, b in intervals)
    changes = Counter()
    for a, b in intervals:
        changes[a] += 1
        changes[b] -= 1
    active = peak = integral = 0
    previous = first
    histogram = Counter()
    for t, change in sorted(changes.items()):
        histogram[active] += t - previous
        integral += active * (t - previous)
        active += change
        peak = max(peak, active)
        previous = t
    return dict(
        peak_overlapping_intervals=peak,
        time_weighted_overlap=integral / (last - first),
        interval_count=len(intervals),
        overlap_duration_ns={str(n): d for n, d in sorted(histogram.items()) if d},
        common_overlap_fraction=max(
            0, min(b for _, b in intervals) - max(a for a, _ in intervals)
        )
        / (last - first),
    )


def audit_schedule(root):
    root = Path(root)
    # Verify all cases, raw hashes, numerical results, geometry and monitoring
    # before interpreting any interval or stored SM observation.
    audit_grid(root)
    rows = []
    for case in footprint_grid():
        p, s = case["programs"], case["local_slots"]
        stem = f"local_s{s}_p{p}_i65536"
        record = json.loads((root / (stem + ".json")).read_text())
        observations = record["native_audit"].get("sm_observations")
        intervals = [
            tuple(map(int, line.split(",")[3:5]))
            for line in (root / (stem + ".log")).read_text().splitlines()
            if line.startswith("body,")
        ]
        row = dict(**case, global_overlap=interval_overlap(intervals))
        if observations is None:
            row.update(sm_observed=False, per_sm=None)
        else:
            changed = [
                i
                for i, obs in enumerate(observations)
                if obs["start_sm"] != obs["end_sm"]
            ]
            by_sm = {}
            for interval, obs in zip(intervals, observations):
                by_sm.setdefault(obs["start_sm"], []).append(interval)
            row.update(
                sm_observed=True,
                changed_endpoint_ctas=changed,
                start_sm_counts={
                    str(sm): len(values) for sm, values in sorted(by_sm.items())
                },
                per_sm=None
                if changed
                else {
                    str(sm): interval_overlap(values)
                    for sm, values in sorted(by_sm.items())
                },
            )
        rows.append(row)
    return dict(
        role="control",
        eligible_for_fit=False,
        rows=rows,
        caveat="Endpoint agreement does not prove absence of preemption/migration inside intervals. Instrumented profiler replay is not a latency target or an issue/residency trace. No points removed.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Require fresh output")
    _write(args.output, audit_schedule(args.root))


if __name__ == "__main__":
    main()
