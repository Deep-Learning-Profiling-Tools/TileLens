from collections import Counter

import pytest

from triton_viz.tools.gpu_cache_counter_model_audit import (
    audit,
    copy_range_accesses,
    copy_range_misses,
)
from triton_viz.performance.gpu_cache import write_allocating_cache_traffic


@pytest.mark.parametrize("tile", [32, 256, 1024])
@pytest.mark.parametrize("ways", [1, 2, 4])
@pytest.mark.parametrize("eviction", ["none", "zero", "read"])
def test_closed_form_matches_explicit_stack(tile, ways, eviction):
    accesses = copy_range_accesses(8192, eviction, l2_bytes=16384, tile_bytes=tile)
    explicit = write_allocating_cache_traffic(
        accesses, capacity_blocks=16384 // tile, associativity=ways
    )
    assert copy_range_misses(
        8192, eviction, l2_bytes=16384, tile_bytes=tile, associativity=ways
    ) == pytest.approx(explicit["load_misses"])


def test_tile_range_preserves_output_pollution_and_sweep_read_traffic():
    plain = copy_range_accesses(8192, "none", l2_bytes=16384)
    zero = copy_range_accesses(8192, "zero", l2_bytes=16384)
    read = copy_range_accesses(8192, "read", l2_bytes=16384)
    assert Counter(op for op, _ in plain) == {"load": 24, "store": 24}
    assert Counter(op for op, _ in zero) == {"load": 24, "store": 56}
    assert Counter(op for op, _ in read) == {"load": 56, "store": 56}
    # Input/output addresses are distinct, but the same input identities recur.
    assert plain[:4] == [("load", ("input", i)) for i in range(4)]
    assert plain[4:8] == [("store", ("output", i)) for i in range(4)]
    assert plain[:16] == plain[16:32] == plain[32:48]


@pytest.mark.parametrize("tile", [0, -1, 1.5, True, 3000])
def test_invalid_tile_units_rejected(tile):
    with pytest.raises(ValueError):
        copy_range_accesses(8192, "none", l2_bytes=16384, tile_bytes=tile)


def test_counter_hypotheses_never_accept_holdout_or_incomplete_matrix():
    with pytest.raises(ValueError, match="control"):
        audit(
            dict(role="holdout", complete=True, replay_counts_consistent=True),
            l2_bytes=25165824,
            associativities=[4],
        )
    with pytest.raises(ValueError, match="nine-control"):
        audit(
            dict(role="control", complete=True, replay_counts_consistent=True, rows=[]),
            l2_bytes=25165824,
            associativities=[4],
        )
