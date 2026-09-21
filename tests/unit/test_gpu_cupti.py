import copy

import pytest

from microbench.gpu.harness.cupti import validate_timestamps
from microbench.gpu.harness.cupti import CuptiTimestamps


def rows():
    return [
        dict(name=name, start_ns=start, end_ns=end, device=0, context=1, stream=2)
        for name, start, end in [("flush", 1, 100), ("target", 101, 110)]
    ]


def test_timestamp_validation_uses_gpu_kernel_intervals():
    result = validate_timestamps(
        rows()[::-1], expected_names=["flush", "target"], device=0
    )
    assert result[1]["end_ns"] - result[1]["start_ns"] == 9


@pytest.mark.parametrize(
    "field,value",
    [("start_ns", 0), ("end_ns", 101), ("device", 1), ("stream", 3), ("name", "other")],
)
def test_timestamp_invalid_batch_is_not_filtered(field, value):
    records = copy.deepcopy(rows())
    records[1][field] = value
    with pytest.raises(ValueError):
        validate_timestamps(records, expected_names=["flush", "target"], device=0)


def test_timestamp_missing_and_extra_records_fail():
    for records in (rows()[:1], rows() + rows()):
        with pytest.raises(ValueError):
            validate_timestamps(records, expected_names=["flush", "target"], device=0)


def test_cupti_waits_for_declared_records_not_a_favorable_duration(monkeypatch):
    class Library:
        flushes = 0

        def tv_cupti_flush(self):
            self.flushes += 1
            return 0

        def tv_cupti_count(self):
            return 2 if self.flushes >= 2 else 0

        def tv_cupti_dropped(self):
            return 0

        def tv_cupti_get(self, index, pointer):
            item = pointer._obj
            item.name = b"flush" if index == 0 else b"target"
            item.start_ns, item.end_ns = 1 + index * 100, 99 + index * 100
            return 0

    collector = object.__new__(CuptiTimestamps)
    collector._lib = Library()
    monkeypatch.setattr("microbench.gpu.harness.cupti.time.sleep", lambda _: None)
    assert len(collector.read(expected_count=2)) == 2
    assert collector._lib.flushes == 2
    with pytest.raises(RuntimeError, match="Timed out"):
        collector.read(expected_count=3, timeout_seconds=0)
    with pytest.raises(ValueError):
        collector.read(expected_count=-1)
    with pytest.raises(ValueError):
        collector.read(timeout_seconds=float("nan"))
