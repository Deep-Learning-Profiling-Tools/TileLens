"""CPU-only tile stack distances and stochastic cache hit probabilities.

This is a modeling primitive, not yet a calibrated GPU cache predictor. Tile
keys must identify equal-sized, non-overlapping source memory blocks across
operands; traversal order is an explicit assumption, not a hardware schedule.
No compiler artifacts, device initialization or fitted kernel constants occur
here. Hardware/cache-policy parameters must come from control calibration.
"""

from __future__ import annotations

import math
from itertools import product


def cyclic_local_cache_hypothesis(
    *,
    actors,
    slots,
    blocks_per_slot,
    iterations,
    capacity_blocks,
    associativity,
    method="exact",
):
    """Closed-form SDCM for a declared round-robin local-array access schedule.

    Each slot has disjoint blocks per actor. Initialize all slots by stores;
    cyclically load then store one slot per actor; finish with one read of all
    slots. Actors interleave at whole-slot boundaries in every phase. Iterations
    must contain complete slot cycles. Residency, per-warp issue order, hardware
    allocation, write policy and cache parameters still need control validation.
    This is not a spill predictor or a calibrated local-memory cache model.
    """
    if any(
        isinstance(x, bool) or not isinstance(x, int) or x < 1
        for x in (actors, slots, blocks_per_slot, iterations)
    ):
        raise ValueError("Require positive integral cyclic schedule dimensions")
    if iterations % slots:
        raise ValueError("Partial cycles require explicit boundary accounting")
    footprint = actors * slots * blocks_per_slot
    load_distance, store_distance = footprint - 1, blocks_per_slot - 1
    load_hit = sdcm_hit_probability(
        load_distance,
        capacity_blocks=capacity_blocks,
        associativity=associativity,
        method=method,
    )
    store_hit = sdcm_hit_probability(
        store_distance,
        capacity_blocks=capacity_blocks,
        associativity=associativity,
        method=method,
    )
    load_requests = actors * blocks_per_slot * (iterations + slots)
    repeated_stores = actors * blocks_per_slot * iterations
    return dict(
        footprint_blocks=footprint,
        load_reuse_distance=load_distance,
        repeated_store_reuse_distance=store_distance,
        traffic=dict(
            load_requests=load_requests,
            load_hits=load_requests * load_hit,
            load_misses=load_requests * (1 - load_hit),
            load_cold_misses=0,
            store_requests=footprint + repeated_stores,
            store_hits=repeated_stores * store_hit,
            store_misses=footprint + repeated_stores * (1 - store_hit),
            store_cold_misses=footprint,
        ),
        calibrated=False,
        schedule="slot_round_robin_all_actors",
        policy="write_allocate_uniform_sets_lru_hypothesis",
    )


def source_cache_traffic(
    source, *, capacity_blocks, associativity, schedule, sm_count, method="exact"
):
    """Evaluate explicit source scheduling hypotheses, never a hardware trace.

    ``program_serial`` executes programs consecutively. ``sm_wave_interleave``
    runs one source memory event per program in round-robin waves of SM count.
    Neither is claimed to bound real miss traffic: residency, issue order and
    overlap must be validated on controls. Canonical 32-byte source sectors are
    the model units; this does not assert a physical cache line size.
    """
    if schedule not in {"program_serial", "sm_wave_interleave"}:
        raise ValueError("Unknown source cache scheduling hypothesis")
    if isinstance(sm_count, bool) or not isinstance(sm_count, int) or sm_count <= 0:
        raise ValueError("A positive hardware SM count is required")
    trace = source["cache_access_trace"]
    if (
        trace.get("schema") != "triton-viz.gpu-source-cache-access.v1"
        or trace.get("block_bytes") != 32
    ):
        raise ValueError("Unsupported source cache trace")
    grid = source["grid"]
    if not grid or any(
        isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in grid
    ):
        raise ValueError("Invalid source launch grid")
    if source["program_count"] != math.prod(grid):
        raise ValueError("Source grid/program count mismatch")
    # Keep zero-memory programs in scheduling waves instead of compressing them
    # away and accidentally advancing later programs into an earlier wave.
    by_program = {index: [] for index in product(*(range(n) for n in grid))}
    seen_blocks = set()
    previous_seq = -1
    for event in trace["accesses"]:
        if event["seq"] <= previous_seq or event["op"] not in {"load", "store"}:
            raise ValueError("Invalid source cache event order or operation")
        previous_seq = event["seq"]
        if any(
            isinstance(block, bool) or not isinstance(block, int) or block < 0
            for block in event["blocks"]
        ):
            raise ValueError("Invalid canonical source block")
        if len(set(event["blocks"])) != len(event["blocks"]):
            raise ValueError("Source event sectors must be unique")
        seen_blocks.update(event["blocks"])
        program = tuple(event["program"])
        if program not in by_program:
            raise ValueError("Source cache event outside launch grid")
        by_program[program].append(event)
    if seen_blocks != set(range(trace["unique_blocks"])):
        raise ValueError("Incomplete canonical source block domain")
    programs = list(by_program.values())
    ordered = []
    width = 1 if schedule == "program_serial" else sm_count
    for start in range(0, len(programs), width):
        wave = programs[start : start + width]
        for index in range(max(len(events) for events in wave)):
            for events in wave:
                if index < len(events):
                    event = events[index]
                    ordered.extend((event["op"], block) for block in event["blocks"])
    return dict(
        traffic=write_allocating_cache_traffic(
            ordered,
            capacity_blocks=capacity_blocks,
            associativity=associativity,
            method=method,
        ),
        schedule=schedule,
        sm_count=sm_count,
        block_bytes=32,
        policy="write_allocate_uniform_sets_lru_hypothesis",
        calibrated=False,
    )


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
    probabilities = {None: 0.0}
    for (op, _), distance in zip(accesses, distances):
        if distance not in probabilities:
            probabilities[distance] = sdcm_hit_probability(
                distance,
                capacity_blocks=capacity_blocks,
                associativity=associativity,
                method=method,
            )
        probability = probabilities[distance]
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
