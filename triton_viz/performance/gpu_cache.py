"""CPU-only tile stack distances and stochastic cache hit probabilities.

This is a modeling primitive, not yet a calibrated GPU cache predictor. Tile
keys must identify equal-sized, non-overlapping source memory blocks across
operands; traversal order is an explicit assumption, not a hardware schedule.
No compiler artifacts, device initialization or fitted kernel constants occur
here. Hardware/cache-policy parameters must come from control calibration.
"""

from __future__ import annotations

import math


def write_allocating_cache_traffic(
    accesses, *, capacity_blocks, associativity, method="exact"
):
    """Evaluate an explicit load/store block order under a write-allocate model.

    Each access is ("load" or "store", canonical_block_key). Stores participate
    in stack distance and compete with loads. This is a policy hypothesis to
    validate on controls, not a claim about GPU replacement or DRAM bytes:
    writeback traffic, write combining and physical execution order are absent.
    """
    accesses = list(accesses)
    if any(op not in {"load", "store"} for op, _ in accesses):
        raise ValueError("Cache accesses must explicitly identify load/store")
    # Validate cache parameters even for an empty sequence.
    sdcm_hit_probability(
        None,
        capacity_blocks=capacity_blocks,
        associativity=associativity,
        method=method,
    )
    distances = tile_stack_distances(key for _, key in accesses)
    result = {
        f"{op}_{metric}": 0.0
        for op in ("load", "store")
        for metric in ("requests", "hits", "misses", "cold_misses")
    }
    for (op, _), distance in zip(accesses, distances):
        probability = sdcm_hit_probability(
            distance,
            capacity_blocks=capacity_blocks,
            associativity=associativity,
            method=method,
        )
        result[f"{op}_requests"] += 1
        result[f"{op}_hits"] += probability
        result[f"{op}_misses"] += 1 - probability
        result[f"{op}_cold_misses"] += distance is None
    return result


def tile_stack_distances(keys):
    """Count distinct blocks since the last access; None denotes a cold miss.

    A Fenwick tree tracks the last occurrence of each block in O(n log n).
    Aliased source regions must share keys; operand names alone are not keys.
    Each call starts cold. Returned values retain every access, including first
    touches and immediate reuse (distance zero).
    """
    keys = list(keys)
    tree = [0] * (len(keys) + 1)
    last = {}
    result = []

    def add(index, value):
        while index < len(tree):
            tree[index] += value
            index += index & -index

    def prefix(index):
        total = 0
        while index:
            total += tree[index]
            index -= index & -index
        return total

    for index, key in enumerate(keys, 1):
        previous = last.get(key)
        result.append(None if previous is None else len(last) - prefix(previous))
        if previous is not None:
            add(previous, -1)
        add(index, 1)
        last[key] = index
    return result


def sdcm_hit_probability(distance, *, capacity_blocks, associativity, method="exact"):
    """P(Binomial(distance, associativity/capacity) < associativity).

    Assumes independent uniform set mapping and LRU-like replacement. Exact
    binomial evaluation is the reference; the Gaussian path uses a signed,
    continuity-corrected normal CDF, with exact deterministic boundary cases.
    Real GPU replacement/partition behavior requires control validation.
    """
    for name, value in (
        ("capacity_blocks", capacity_blocks),
        ("associativity", associativity),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if capacity_blocks < associativity or capacity_blocks % associativity:
        raise ValueError("Capacity must contain an integral positive number of sets")
    if method not in {"exact", "gaussian"}:
        raise ValueError("Unknown SDCM method")
    if distance is None:
        return 0.0
    if isinstance(distance, bool) or not isinstance(distance, int) or distance < 0:
        raise ValueError("Distance must be a nonnegative integer or None")
    if distance < associativity:
        return 1.0
    p = associativity / capacity_blocks
    if p == 1:
        return 0.0
    if method == "gaussian":
        mean = distance * p
        variance = mean * (1 - p)
        z = (associativity - 0.5 - mean) / math.sqrt(variance)
        return 0.5 * math.erfc(-z / math.sqrt(2))
    terms = [
        math.exp(
            math.lgamma(distance + 1)
            - math.lgamma(k + 1)
            - math.lgamma(distance - k + 1)
            + k * math.log(p)
            + (distance - k) * math.log1p(-p)
        )
        for k in range(associativity)
    ]
    return min(1.0, max(0.0, math.fsum(terms)))
