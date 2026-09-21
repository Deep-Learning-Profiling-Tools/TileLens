import copy
from unittest.mock import Mock

import pytest

from microbench.gpu.harness.cupti import validate_timestamps
from microbench.gpu.harness.cupti import CuptiTimestamps, group_kernel_intervals


def test_software_diagnostics_require_explicit_selection_and_matching_kind(
    tmp_path, monkeypatch
):
    library = Mock()
    library.tv_cupti_hardware_trace.return_value = 0
    library.tv_cupti_activity_kind.return_value = 3
    library.tv_cupti_init.return_value = 0
    library.tv_cupti_close.return_value = 0
    path = tmp_path / "bridge.so"
    path.write_bytes(b"fake diagnostic bridge")
    monkeypatch.setattr("microbench.gpu.harness.cupti.ctypes.CDLL", lambda _: library)
    with pytest.raises(ValueError, match="non-HES"):
        CuptiTimestamps(path)
    library.tv_cupti_init.assert_not_called()
    probe = CuptiTimestamps(path, timestamp_method="software_serial")
    assert probe.timestamp_method == "software_serial"
    probe.close()
    library.tv_cupti_activity_kind.return_value = 10
    with pytest.raises(ValueError, match="labelled bridge"):
        CuptiTimestamps(path, timestamp_method="software_serial")


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


def test_cupti_explicit_cleanup_is_idempotent():
    class Library:
        closes = 0

        def tv_cupti_close(self):
            self.closes += 1
            return 0

    collector = object.__new__(CuptiTimestamps)
    collector._lib, collector._closed = Library(), False
    collector.close()
    collector.close()
    assert collector._closed and collector._lib.closes == 1


def test_fixed_group_means_include_slow_intervals_without_filtering():
    records = [
        dict(start_ns=1, end_ns=2),
        dict(start_ns=3, end_ns=1003),
        dict(start_ns=1004, end_ns=1005),
        dict(start_ns=1006, end_ns=1001006),
    ]
    samples = group_kernel_intervals(records, kernels_per_sample=2, sample_count=1)
    assert samples[0]["records"] == records
    assert samples[0]["kernel_latencies_us"] == [1, 1000]
    assert samples[0]["latency_us"] == 500.5
    with pytest.raises(ValueError):
        group_kernel_intervals(records[:-1], kernels_per_sample=2, sample_count=1)


@pytest.mark.parametrize("chunk", ["0", "-1", "3", "12"])
def test_invalid_graph_chunks_rejected_before_gpu_access(tmp_path, chunk):
    from triton_viz.tools.gpu_cupti_probe import main

    with pytest.raises(ValueError, match="Graph chunk"):
        main(
            [
                "--library",
                str(tmp_path / "absent.so"),
                "--output",
                str(tmp_path / "probe.json"),
                "--graph-samples",
                "--graph-pairs-per-replay",
                chunk,
            ]
        )


def test_post_teardown_snapshot_never_flushes_or_filters_incomplete_records():
    class Library:
        def tv_cupti_count(self):
            return 1

        def tv_cupti_get(self, index, pointer):
            pointer._obj.name = b"incomplete"
            pointer._obj.start_ns = 0
            pointer._obj.end_ns = 0
            return 0

    collector = object.__new__(CuptiTimestamps)
    collector._lib = Library()
    result = collector.snapshot()
    assert len(result) == 1 and result[0]["start_ns"] == 0
    with pytest.raises(ValueError, match="timestamps"):
        validate_timestamps(result, expected_names=["incomplete"], device=0)
