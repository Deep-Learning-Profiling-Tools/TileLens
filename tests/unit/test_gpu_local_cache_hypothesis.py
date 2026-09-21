import pytest

from triton_viz.performance.gpu_cache import (
    cyclic_local_cache_hypothesis,
    write_allocating_cache_traffic,
)


@pytest.mark.parametrize("actors,slots,blocks", [(1, 2, 1), (2, 4, 2), (3, 2, 3)])
@pytest.mark.parametrize("method", ["exact", "gaussian"])
def test_closed_form_matches_every_explicit_access(actors, slots, blocks, method):
    iterations = slots * 2
    accesses = []
    for slot in range(slots):
        for actor in range(actors):
            accesses.extend(("store", (actor, slot, b)) for b in range(blocks))
    for iteration in range(iterations):
        for actor in range(actors):
            for op in ("load", "store"):
                accesses.extend(
                    (op, (actor, iteration % slots, b)) for b in range(blocks)
                )
    for slot in range(slots):
        for actor in range(actors):
            accesses.extend(("load", (actor, slot, b)) for b in range(blocks))
    for capacity in (4, 16, 64):
        explicit = write_allocating_cache_traffic(
            accesses, capacity_blocks=capacity, associativity=2, method=method
        )
        predicted = cyclic_local_cache_hypothesis(
            actors=actors,
            slots=slots,
            blocks_per_slot=blocks,
            iterations=iterations,
            capacity_blocks=capacity,
            associativity=2,
            method=method,
        )
        assert predicted["traffic"] == pytest.approx(explicit)
        assert predicted["calibrated"] is False


def test_partial_cycles_and_unknown_residency_are_not_silently_guessed():
    with pytest.raises(ValueError, match="Partial"):
        cyclic_local_cache_hypothesis(
            actors=1,
            slots=128,
            blocks_per_slot=16,
            iterations=16,
            capacity_blocks=1024,
            associativity=4,
        )
    with pytest.raises(ValueError, match="positive"):
        cyclic_local_cache_hypothesis(
            actors=0,
            slots=128,
            blocks_per_slot=16,
            iterations=65536,
            capacity_blocks=1024,
            associativity=4,
        )
