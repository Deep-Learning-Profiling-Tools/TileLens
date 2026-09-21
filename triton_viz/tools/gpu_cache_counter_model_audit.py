"""Compare explicit tile-SDCM hypotheses with the fixed cache control matrix.

No parameter selection or latency fitting. All associativity scenarios and all
declared controls remain visible. The 1 KiB block divides both declared copy tile
sizes; it is a source tile unit, not an assertion about L2 physical line size.
"""

import argparse
import json
import math
from pathlib import Path

from triton_viz.performance.gpu_cache import sdcm_hit_probability
from triton_viz.tools.gpu_cost_model_pipeline import _write
from microbench.gpu.common.cache_controls import cache_declaration


def copy_range_accesses(working_set_bytes, eviction, *, l2_bytes, tile_bytes=1024):
    if eviction not in {"none", "zero", "read"}:
        raise ValueError("Unknown declared eviction")
    if any(
        isinstance(n, bool) or not isinstance(n, int) or n <= 0
        for n in (working_set_bytes, l2_bytes, tile_bytes)
    ):
        raise ValueError("Positive integral memory sizes are required")
    if (
        any(n <= 0 or n % tile_bytes for n in (working_set_bytes, l2_bytes))
        or 4096 % tile_bytes
    ):
        raise ValueError("Working sets and copy tiles must contain whole model blocks")
    blocks = working_set_bytes // tile_bytes
    copy_width = 4096 // tile_bytes  # Declared 1024 fp32 elements per program.
    accesses = []
    for launch in range(3):
        if launch == 2 and eviction != "none":
            for index in range(2 * l2_bytes // tile_bytes):
                if eviction == "read":
                    accesses.append(("load", ("sweep_in", index)))
                    accesses.append(("store", ("sweep_out", index)))
                else:
                    accesses.append(("store", ("sweep_in", index)))
        for first in range(0, blocks, copy_width):
            for op, operand in (("load", "input"), ("store", "output")):
                accesses.extend(
                    (op, (operand, index))
                    for index in range(first, min(first + copy_width, blocks))
                )
    return accesses


def copy_range_misses(
    working_set_bytes,
    eviction,
    *,
    l2_bytes,
    associativity,
    tile_bytes=1024,
    method="exact",
):
    """Closed-form load misses for the declared repeated copy sequence.

    Each reused input block encounters every other input/output block between
    uses. A fresh sweep inserts one (zero) or two (read/copy) distinct operands.
    This is exactly the serial source trace hypothesis, not a GPU schedule fit.
    """
    # Validate sizes without allocating a working-set-sized trace.
    copy_range_accesses(
        tile_bytes, eviction, l2_bytes=tile_bytes, tile_bytes=tile_bytes
    )
    if any(
        isinstance(n, bool) or not isinstance(n, int) or n <= 0 or n % tile_bytes
        for n in (working_set_bytes, l2_bytes)
    ):
        raise ValueError("Working sets must contain positive whole model blocks")
    blocks = working_set_bytes // tile_bytes
    capacity = l2_bytes // tile_bytes
    sweep = 0 if eviction == "none" else 2 * capacity
    distance = 2 * blocks - 1

    def miss(d):
        return 1 - sdcm_hit_probability(
            d, capacity_blocks=capacity, associativity=associativity, method=method
        )

    return blocks * (
        1 + miss(distance) + miss(distance + sweep * (2 if eviction == "read" else 1))
    ) + (sweep if eviction == "read" else 0)


def audit(counter_report, *, l2_bytes, associativities, tile_bytes=1024):
    if (
        counter_report.get("role") != "control"
        or not counter_report.get("complete")
        or not counter_report.get("replay_counts_consistent")
    ):
        raise ValueError("Require complete consistent control range counters")
    if not associativities or len(set(associativities)) != len(associativities):
        raise ValueError("Declare nonempty distinct associativity scenarios")
    matrix = counter_report.get("matrix", "legacy")
    declaration = cache_declaration(matrix)
    expected = {
        (mib, eviction)
        for mib in declaration["working_set_mib"]
        for eviction in declaration["evictions"]
    }
    rows = counter_report["rows"]
    if (
        len(rows) != len(expected)
        or {(r["working_set_mib"], r["eviction"]) for r in rows} != expected
    ):
        raise ValueError(
            "Keep the exact declared nine-control matrix"
            if matrix == "legacy"
            else "Keep the exact declared capacity-control matrix"
        )
    if l2_bytes != declaration["l2_bytes"]:
        raise ValueError("Counter and model hardware capacities must match")
    scenarios = []
    for ways in associativities:
        comparisons = []
        for row in rows:
            sectors = copy_range_misses(
                row["working_set_mib"] * 1024 * 1024,
                row["eviction"],
                l2_bytes=l2_bytes,
                associativity=ways,
                tile_bytes=tile_bytes,
            ) * (tile_bytes / 32)
            measured = row["counters"]["misses"]
            if not math.isfinite(measured) or measured <= 0:
                raise ValueError("Require positive measured control miss counts")
            comparisons.append(
                dict(
                    working_set_mib=row["working_set_mib"],
                    eviction=row["eviction"],
                    predicted_read_miss_sectors=sectors,
                    measured_read_miss_sectors=measured,
                    relative_error_pct=100 * (sectors / measured - 1),
                )
            )
        scenarios.append(
            dict(
                associativity=ways,
                rows=comparisons,
                miss_count_mape_pct=sum(
                    abs(r["relative_error_pct"]) for r in comparisons
                )
                / len(comparisons),
            )
        )
    return dict(
        role="control",
        matrix=matrix,
        eligible_for_fit=False,
        l2_bytes=l2_bytes,
        tile_bytes=tile_bytes,
        schedule="serial CTA source tile bursts across three launches",
        scenarios=scenarios,
        selected=None,
        caveat="Hypothesis sensitivity, not cache calibration or latency CV; stores allocate, eviction-read traffic is included, cache lines/sets and physical scheduling are not identified.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counters", type=Path, required=True)
    parser.add_argument("--l2-bytes", type=int, required=True)
    parser.add_argument("--tile-bytes", type=int, default=1024)
    parser.add_argument("--associativities", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Use a new audit output")
    result = audit(
        json.loads(args.counters.read_text()),
        l2_bytes=args.l2_bytes,
        associativities=args.associativities,
        tile_bytes=args.tile_bytes,
    )
    _write(args.output, result)
    print([(r["associativity"], r["miss_count_mape_pct"]) for r in result["scenarios"]])


if __name__ == "__main__":
    main()
