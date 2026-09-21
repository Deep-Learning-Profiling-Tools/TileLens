import os

import pytest

from microbench.gpu.harness import measure


@pytest.mark.parametrize("foreign", [False, True])
def test_monitor_retains_raw_results_and_foreign_process_rejections(
    monkeypatch, foreign
):
    calls = 0

    def sample(device):
        nonlocal calls
        calls += 1
        return dict(
            processes=[dict(pid=-1 if foreign and calls > 1 else os.getpid())],
            graphics_processes=[dict(pid=10)],
            utilization_pct=100,
        )

    monkeypatch.setattr(measure, "snapshot", sample)
    raw, monitor = measure.monitor_call(
        lambda: [dict(start_ns=1, end_ns=10)], allowed_graphics=(10,)
    )
    assert raw == [dict(start_ns=1, end_ns=10)]
    assert monitor["contaminated"] is foreign
    assert len(monitor["samples"]) >= 2
    assert monitor["isolation"] == "monitored_shared_desktop"


def test_monitor_query_failure_rejects_measurement(monkeypatch):
    def unavailable(*args):
        raise RuntimeError("NVML unavailable")

    monkeypatch.setattr(measure, "snapshot", unavailable)
    with pytest.raises(measure.BusyGPU, match="Monitoring rejected"):
        measure.monitor_call(lambda: pytest.fail("Must not launch without monitoring"))
