import copy

import pytest

from triton_viz.performance.gpu_cache import source_cache_traffic


def source():
    return dict(
        grid=[2],
        program_count=2,
        cache_access_trace=dict(
            schema="triton-viz.gpu-source-cache-access.v1",
            block_bytes=32,
            unique_blocks=2,
            accesses=[
                dict(seq=i, program=[program], op="load", blocks=[block])
                for i, (program, block) in enumerate([(0, 0), (0, 0), (1, 1), (1, 1)])
            ],
        ),
    )


def test_schedule_hypotheses_change_reuse_without_compiler_inputs():
    trace = source()
    frozen = copy.deepcopy(trace)
    serial = source_cache_traffic(
        trace, capacity_blocks=1, associativity=1, sm_count=2, schedule="program_serial"
    )
    waves = source_cache_traffic(
        trace,
        capacity_blocks=1,
        associativity=1,
        sm_count=2,
        schedule="sm_wave_interleave",
    )
    assert serial["traffic"]["load_misses"] == 2
    assert waves["traffic"]["load_misses"] == 4
    assert serial["traffic"]["load_requests"] == waves["traffic"]["load_requests"] == 4
    assert not serial["calibrated"] and not waves["calibrated"]
    assert trace == frozen


def test_stores_and_cross_program_aliases_share_stack():
    trace = source()
    trace["cache_access_trace"]["accesses"][0]["op"] = "store"
    result = source_cache_traffic(
        trace,
        capacity_blocks=1,
        associativity=1,
        sm_count=1,
        schedule="sm_wave_interleave",
    )["traffic"]
    assert result["store_misses"] == 1 and result["load_hits"] == 2


def test_empty_programs_remain_in_waves():
    trace = source()
    trace["grid"], trace["program_count"] = [3], 3
    for event in trace["cache_access_trace"]["accesses"]:
        if event["program"] == [1]:
            event["program"] = [2]
    result = source_cache_traffic(
        trace,
        capacity_blocks=1,
        associativity=1,
        sm_count=2,
        schedule="sm_wave_interleave",
    )
    assert result["traffic"]["load_misses"] == 2


@pytest.mark.parametrize(
    "change", ["duplicate", "missing_block", "bad_op", "bad_order"]
)
def test_malformed_cache_trace_rejected(change):
    trace = source()
    event = trace["cache_access_trace"]["accesses"][0]
    if change == "duplicate":
        event["blocks"] = [0, 0]
    elif change == "missing_block":
        event["blocks"] = [3]
    elif change == "bad_op":
        event["op"] = "unknown"
    else:
        event["seq"] = 5
    with pytest.raises(ValueError):
        source_cache_traffic(
            trace,
            capacity_blocks=1,
            associativity=1,
            sm_count=2,
            schedule="program_serial",
        )
