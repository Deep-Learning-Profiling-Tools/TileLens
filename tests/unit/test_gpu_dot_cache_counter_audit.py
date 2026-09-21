from pathlib import Path

import pytest

from triton_viz.tools.gpu_dot_cache_counter_audit import parse
from triton_viz.tools.gpu_local_counter_collect import CACHE_METRICS, commands


def csv(values):
    return "\n".join(
        ['"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"']
        + [
            f'"0","geometry_dot","{name}","sector","{value}"'
            for name, value in zip(CACHE_METRICS, values)
        ]
    )


def test_zero_spill_controls_and_load_store_asymmetry_remain_explicit():
    zero = parse(csv([0] * 9))
    assert zero["LDL"]["hit_fraction"] is None
    assert zero["STL"]["misses"] == 0
    result = parse(csv([100, 100, 1, 99, 100, 0, 110, 20, 89]))
    assert result["LDL"]["hit_fraction"] == 0.01
    assert result["STL"]["hit_fraction"] == 1
    assert result["L2_read"]["replay_count_difference"] == -1
    with pytest.raises(ValueError, match="Incomplete"):
        parse(csv([0] * 8))
    with pytest.raises(ValueError, match="integer"):
        parse(csv([-1] + [0] * 8))
    with pytest.raises(ValueError, match="Unexpected"):
        parse(csv([0] * 9).replace('"sector"', '"byte"'))


@pytest.mark.parametrize(
    "suite,count", [("pressure", 32), ("pressure_pipeline", 32), ("resource_dot", 96)]
)
def test_cache_phase_preserves_full_suite_and_cold_replay(suite, count):
    result = commands("ncu", Path("output"), suite=suite, cache_lookups=True)
    assert len(result) == count
    for command in result:
        assert command[command.index("--metrics") + 1] == ",".join(CACHE_METRICS)
        assert command[command.index("--cache-control") + 1] == "all"
        assert command[command.index("--clock-control") + 1] == "none"
