import math
import random

import pytest

from triton_viz.performance.gpu_cache import sdcm_hit_probability, tile_stack_distances
from triton_viz.performance.gpu_cache import write_allocating_cache_traffic


def test_tile_stack_distances_keep_cold_and_repeated_accesses():
    assert tile_stack_distances([]) == []
    assert tile_stack_distances("abacba") == [None, None, 1, None, 2, 2]
    assert tile_stack_distances("aaaa") == [None, 0, 0, 0]


def test_stack_distances_match_explicit_lru_stack():
    rng = random.Random(0)
    keys = [rng.randrange(32) for _ in range(1000)]
    stack, expected = [], []
    for key in keys:
        expected.append(stack.index(key) if key in stack else None)
        if key in stack:
            stack.remove(key)
        stack.insert(0, key)
    assert tile_stack_distances(iter(keys)) == expected


@pytest.mark.parametrize("method", ["exact", "gaussian"])
def test_sdcm_boundaries_and_monotonicity(method):
    def hit(distance, capacity=128, ways=8):
        return sdcm_hit_probability(
            distance, capacity_blocks=capacity, associativity=ways, method=method
        )

    assert hit(None) == 0
    assert hit(0) == 1
    assert hit(7, 8) == 1
    assert hit(8, 8) == 0
    values = [hit(d) for d in range(1024)]
    assert all(0 <= v <= 1 for v in values)
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert hit(100000) < 1e-10
    assert hit(64, 256) >= hit(64, 128)


def test_exact_binomial_reference_and_gaussian_signed_tail():
    for distance in range(24):
        expected = sum(
            math.comb(distance, k) * 0.25**k * 0.75 ** (distance - k)
            for k in range(min(4, distance + 1))
        )
        assert sdcm_hit_probability(
            distance, capacity_blocks=16, associativity=4
        ) == pytest.approx(expected)
    # A normal CDF must tend to zero, not one, as reuse distance grows.
    assert (
        sdcm_hit_probability(
            4096, capacity_blocks=128, associativity=8, method="gaussian"
        )
        < 1e-20
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(distance=-1),
        dict(distance=1.5),
        dict(distance=True),
        dict(capacity_blocks=0),
        dict(capacity_blocks=9),
        dict(associativity=0),
        dict(associativity=256),
        dict(method="unknown"),
    ],
)
def test_invalid_sdcm_inputs_fail_closed(kwargs):
    params = dict(distance=8, capacity_blocks=128, associativity=8)
    params.update(kwargs)
    with pytest.raises(ValueError):
        sdcm_hit_probability(**params)


def test_stores_compete_for_cache_capacity_and_alias_keys_are_shared():
    kwargs = dict(capacity_blocks=2, associativity=2)
    baseline = write_allocating_cache_traffic(
        [("load", "a"), ("load", "b"), ("load", "a")], **kwargs
    )
    with_store = write_allocating_cache_traffic(
        [("load", "a"), ("load", "b"), ("store", "out"), ("load", "a")], **kwargs
    )
    assert baseline["load_hits"] == 1
    assert with_store["load_hits"] == 0
    assert with_store["load_misses"] == 3
    assert with_store["load_cold_misses"] == 2
    alias = write_allocating_cache_traffic([("store", "a"), ("load", "a")], **kwargs)
    assert alias["load_hits"] == 1 and alias["store_cold_misses"] == 1
    assert all(v == 0 for v in write_allocating_cache_traffic([], **kwargs).values())
    with pytest.raises(ValueError):
        write_allocating_cache_traffic([("unknown", "a")], **kwargs)
