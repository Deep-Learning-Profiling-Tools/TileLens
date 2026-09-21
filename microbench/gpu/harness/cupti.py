"""Explicit control-only HES timestamp collection; no CUDA-event fallback.

Load the separately built CUPTI 13.1 bridge before CUDA context creation.
Hardware timestamps exclude flush kernels and host launch gaps by construction.
An eviction sweep is not proof of cold L2: validate it with control-only counters
before admitting any timings to a cold-cache calibration dataset.
"""

from __future__ import annotations

import ctypes
import math
import time
import statistics
from pathlib import Path


class _Timestamp(ctypes.Structure):
    _fields_ = [
        ("start_ns", ctypes.c_uint64),
        ("end_ns", ctypes.c_uint64),
        ("device", ctypes.c_uint32),
        ("context", ctypes.c_uint32),
        ("stream", ctypes.c_uint32),
        ("correlation", ctypes.c_uint32),
        ("name", ctypes.c_char * 256),
    ]


def validate_timestamps(records, *, expected_names, device):
    """Require every expected launch, in order, without filtering slow samples."""
    ordered = sorted(records, key=lambda r: r["start_ns"])
    if [r["name"] for r in ordered] != list(expected_names):
        raise ValueError("Missing, extra or reordered kernel activity records")
    for row in ordered:
        if row["device"] != device:
            raise ValueError("Unexpected GPU in activity records")
        if row["start_ns"] <= 0 or row["end_ns"] <= row["start_ns"]:
            raise ValueError("Missing or invalid GPU kernel timestamps")
    if len({(r["context"], r["stream"]) for r in ordered}) > 1:
        raise ValueError("Expected a single ordered measurement stream")
    if any(a["end_ns"] > b["start_ns"] for a, b in zip(ordered, ordered[1:])):
        raise ValueError("Overlapping flush/target launches")
    return ordered


def group_kernel_intervals(records, *, kernels_per_sample, sample_count=11):
    """Arithmetic means of fixed-size groups; retain every cold-kernel interval."""
    for n in (kernels_per_sample, sample_count):
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("Positive integral replication counts are required")
    if len(records) != 2 * kernels_per_sample * sample_count:
        raise ValueError("Missing or extra sweep/control records")
    result = []
    for start in range(0, len(records), 2 * kernels_per_sample):
        group = records[start : start + 2 * kernels_per_sample]
        latencies = [(r["end_ns"] - r["start_ns"]) / 1000 for r in group[1::2]]
        if any(not math.isfinite(t) or t <= 0 for t in latencies):
            raise ValueError("Invalid hardware kernel interval")
        result.append(
            dict(
                records=group,
                kernel_latencies_us=latencies,
                latency_us=statistics.mean(latencies),
            )
        )
    return result


class CuptiTimestamps:
    def __init__(self, library, *, delivery_timeout_seconds=1.0):
        if not math.isfinite(delivery_timeout_seconds) or delivery_timeout_seconds < 0:
            raise ValueError("HES delivery timeout must be finite and nonnegative")
        self._delivery_timeout_seconds = delivery_timeout_seconds
        self.library = str(Path(library).resolve(strict=True))
        self._lib = ctypes.CDLL(self.library)
        self._lib.tv_cupti_init.restype = ctypes.c_int
        self._lib.tv_cupti_flush.restype = ctypes.c_int
        self._lib.tv_cupti_count.restype = ctypes.c_size_t
        self._lib.tv_cupti_dropped.restype = ctypes.c_size_t
        self._lib.tv_cupti_get.argtypes = [ctypes.c_size_t, ctypes.POINTER(_Timestamp)]
        self._lib.tv_cupti_get.restype = ctypes.c_int
        self._lib.tv_cupti_clear.restype = None
        self._lib.tv_cupti_close.restype = ctypes.c_int
        self._check(self._lib.tv_cupti_init())
        self._closed = False

    @staticmethod
    def _check(status):
        if status:
            raise RuntimeError(
                f"CUPTI HES bridge failed (status {status}); no fallback"
            )

    def read(self, *, expected_count=None, timeout_seconds=None):
        if timeout_seconds is None:
            timeout_seconds = getattr(self, "_delivery_timeout_seconds", 1.0)
        if expected_count is not None and (
            isinstance(expected_count, bool)
            or not isinstance(expected_count, int)
            or expected_count < 0
        ):
            raise ValueError("Expected record count must be a nonnegative integer")
        if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
            raise ValueError("HES delivery timeout must be finite and nonnegative")
        # GPU completion can precede delivery of HES records. Wait for the
        # declared launch count, not for a timing value or a favorable sample.
        deadline = time.monotonic() + timeout_seconds
        while True:
            self._check(self._lib.tv_cupti_flush())
            if expected_count is None or self._lib.tv_cupti_count() >= expected_count:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Timed out waiting for {expected_count} HES records; received {self._lib.tv_cupti_count()}"
                )
            time.sleep(0.001)
        if self._lib.tv_cupti_dropped():
            raise RuntimeError(
                "CUPTI dropped activity records; reject the entire batch"
            )
        return self.snapshot()

    def snapshot(self):
        """Copy delivered records, including after teardown; no flush or acceptance.

        Teardown may force incomplete records out. Callers must not treat this
        diagnostic snapshot as a validated measurement.
        """
        result = []
        for index in range(self._lib.tv_cupti_count()):
            item = _Timestamp()
            self._check(self._lib.tv_cupti_get(index, ctypes.byref(item)))
            row = {name: getattr(item, name) for name, _ in _Timestamp._fields_}
            row["name"] = row["name"].decode("utf-8", errors="strict")
            result.append(row)
        return result

    def clear(self):
        self.read()  # Check sticky failures before discarding warmup records.
        self._lib.tv_cupti_clear()

    def close(self):
        if not self._closed:
            self._check(self._lib.tv_cupti_close())
            self._closed = True
