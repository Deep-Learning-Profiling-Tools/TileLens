import csv
import io

import pytest

from triton_viz.tools.gpu_local_cache_counter_collect import METRICS, parse_counters
from triton_viz.tools.gpu_cupti_perturbation_collect import _owned_group_sample


def counters():
    buffer = io.StringIO()
    writer = csv.writer(buffer, quoting=csv.QUOTE_ALL)
    writer.writerow(["ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value"])
    for name, value in zip(METRICS, (1000, 1000, 75, 100, 80, 20)):
        writer.writerow(
            [
                0,
                "perturbation_body(float*)",
                name,
                "%" if name.endswith(".pct") else "sector",
                value,
            ]
        )
    return buffer.getvalue()


def test_all_counters_and_replay_disagreement_are_retained():
    result = parse_counters(counters())
    assert len(result["metrics"]) == 6
    assert result["l2_hit_fraction"] == 0.8
    assert result["replay_count_disagreement"] == 0
    result = parse_counters(counters().replace('"100"', '"110"'))
    assert result["replay_count_disagreement"] == pytest.approx(10 / 110)


@pytest.mark.parametrize(
    "old,new",
    [
        ('"0"', '"1"'),
        ('"sector"', '"byte"'),
        ('"75"', '"101"'),
        ('"80"', '"nan"'),
        (METRICS[0], "unknown"),
    ],
)
def test_wrong_launch_units_or_nonfinite_values_are_rejected(old, new):
    with pytest.raises(ValueError):
        parse_counters(counters().replace(old, new))


def test_owned_profiler_group_does_not_whitelist_unrelated_gpu_process(monkeypatch):
    import os

    monkeypatch.setattr(os, "getpgid", lambda pid: {20: 10, 30: 30}[pid])
    raw = dict(processes=[dict(pid=20), dict(pid=30)])
    classified = _owned_group_sample(raw, 10)
    assert classified["processes"] == [dict(pid=10), dict(pid=30)]
    assert raw["processes"] == [dict(pid=20), dict(pid=30)]
    assert raw["observed_process_groups"] == {20: 10, 30: 30}
