from pathlib import Path

import pytest

from triton_viz.tools.gpu_dot_cache_counter_audit import parse, parse_issue
from triton_viz.tools.gpu_local_counter_collect import (
    CACHE_METRICS,
    ISSUE_METRICS,
    commands,
)


def test_instruction_phase_has_typed_counters_not_profile_timing():
    text = "\n".join(
        ['"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"']
        + [
            f'"0","geometry_dot","{name}","{unit}","{value}"'
            for name, unit, value in zip(
                ISSUE_METRICS,
                ["inst"] * 4 + ["%"] * 4,
                [0, 0, 100, 110, 10.5, 20, 30, 40],
            )
        ]
    )
    result = parse_issue(text)
    composed = text.replace('"geometry_dot"', '"composed_dot"')
    assert parse_issue(composed, kernel_name="composed_dot") == result
    with pytest.raises(ValueError, match="Unexpected"):
        parse_issue(composed)
    assert result[ISSUE_METRICS[0]] == 0
    assert result[ISSUE_METRICS[4]] == 10.5
    with pytest.raises(ValueError, match="Invalid"):
        parse_issue(text.replace('"10.5"', '"101"'))
    retained = parse_issue(
        text.replace('"10.5"', '"107.14"'), retain_invalid_stalls=True
    )
    assert retained[ISSUE_METRICS[4]] == 107.14
    with pytest.raises(ValueError, match="Invalid"):
        parse_issue(text.replace('"10.5"', '"nan"'), retain_invalid_stalls=True)
    with pytest.raises(ValueError, match="Unexpected"):
        parse_issue(text.replace('"inst"', '"cycle"'))
    for command in commands("ncu", Path("output"), issue_work=True):
        assert command[command.index("--metrics") + 1] == ",".join(ISSUE_METRICS)
        assert command[command.index("--cache-control") + 1] == "all"
    with pytest.raises(ValueError, match="separate"):
        commands("ncu", Path("output"), issue_work=True, cache_lookups=True)


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
    "suite,count",
    [
        ("pressure", 32),
        ("pressure_pipeline", 32),
        ("resource_dot", 96),
        ("resource_composition", 96),
    ],
)
def test_cache_phase_preserves_full_suite_and_cold_replay(suite, count):
    result = commands("ncu", Path("output"), suite=suite, cache_lookups=True)
    assert len(result) == count
    for command in result:
        assert command[command.index("--metrics") + 1] == ",".join(CACHE_METRICS)
        assert command[command.index("--cache-control") + 1] == "all"
        assert command[command.index("--clock-control") + 1] == "none"


def test_composition_phase_preserves_all_declared_controls_and_kernel_identity():
    from microbench.gpu.common.cases import load_cases
    from triton_viz.tools.gpu_control_resources import selected_controls

    cases = selected_controls("resource_composition")
    assert cases == [
        c
        for c in load_cases("resource_transfer", "control")
        if c["kind"] == "structure_composition"
    ]
    assert len(cases) == 96
    assert load_cases("resource_composition", "holdout") == []
    assert (
        len(
            commands(
                "ncu", Path("output"), suite="resource_composition", issue_work=True
            )
        )
        == 96
    )
    with pytest.raises(ValueError, match="requires"):
        commands("ncu", Path("output"), suite="resource_composition")
    text = csv([0] * 9).replace('"geometry_dot"', '"composed_dot"')
    assert parse(text, kernel_name="composed_dot")["LDL"]["requests"] == 0
    with pytest.raises(ValueError, match="Unexpected"):
        parse(text)
