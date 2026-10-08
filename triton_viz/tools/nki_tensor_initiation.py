"""Experimental NeuronCore-v2 initiation/readiness response.

The architecture's MM initiation rule is distinct from output completion.
Completion tails below are empirical compound response parameters, not proven
native service times. Background LoadStationary is approximated by a maximum
of LS and MM initiation demands. Geometry/sequence transport needs validation.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
import statistics

from triton_viz.tools.nki_tensor_tile_geometry import TensorTileGeometryCalibration


@dataclass(frozen=True)
class TensorInitiationCalibration:
    """Explicit clock and dtype-specific (startup, readiness-tail) nanoseconds."""

    tensor_clock_ghz: float
    points: dict[str, tuple[float, float]]
    mm_minimum_cycles: int = 64
    fast_stationary_ratio: float = 4.

    def __post_init__(self):
        if (not math.isfinite(self.tensor_clock_ghz) or self.tensor_clock_ghz <= 0
                or self.mm_minimum_cycles <= 0 or self.fast_stationary_ratio <= 0
                or not self.points):
            raise ValueError("positive clock, initiation constants and calibration required")
        for dtype, values in self.points.items():
            if dtype not in {"float32", "bfloat16", "float16", "tfloat32"}:
                raise ValueError("unsupported Tensor operand dtype")
            if len(values) != 2 or any(not math.isfinite(v) or v < 0 for v in values):
                raise ValueError("finite nonnegative startup and readiness tail required")

    def timing(self, event):
        if event.get("op") == "dot":
            # Reuse normalized shape/dtype validation, not its tile costs.
            geometry = TensorTileGeometryCalibration(
                {d:(0., 1., 0., 0., 0.) for d in self.points}).geometry(event)
            m, k, n = geometry["shape"]
            dtype = geometry["dtype"]
        elif event.get("op") == "tensor_transpose":
            shape = event.get("input_shape")
            if (not isinstance(shape, (list, tuple)) or len(shape) != 2
                    or any(type(v) is not int or not 0 < v <= 128 for v in shape)):
                raise ValueError("explicit supported Tensor transpose shape required")
            k, m = shape
            n = k  # Transpose = data.T @ identity(K,K).
            types = event.get("input_dtypes") or [event.get("output_dtype")]
            if not types or len(set(types)) != 1 or types[0] not in self.points:
                raise ValueError("explicit calibrated Tensor transpose dtype required")
            dtype = types[0]
        else:
            raise ValueError("Tensor initiation only supports Dot and Tensor transpose")
        if not (0 < m <= 128 and 0 < k <= 128 and 0 < n <= 512):
            raise ValueError("explicit single-NeuronCore-v2-tile geometry required")
        precision = 4. if dtype == "float32" else 1.
        ls = precision*m/self.fast_stationary_ratio/self.tensor_clock_ghz
        mm = precision*max(n, self.mm_minimum_cycles)/self.tensor_clock_ghz
        issue = max(ls, mm)
        tail = self.points[dtype][1]
        return dict(dtype=dtype, ls_initiation_ns=ls, mm_initiation_ns=mm,
                    initiation_ns=issue, completion_ns=issue+tail,
                    readiness_tail_ns=tail, native_service_identified=False)

    def assign_events(self, events):
        types = set()
        count = 0
        for event in events:
            if event.get("op") not in {"dot", "tensor_transpose"}:
                continue
            timing = self.timing(event)
            event["tensor_pipeline_timing"] = timing
            event["scheduler_duration_override_ns"] = timing["initiation_ns"]
            event["tensor_pipeline_ready_tail_ns"] = timing["readiness_tail_ns"]
            types.add(timing["dtype"])
            count += 1
        return dict(startup_ns=sum(self.points[d][0] for d in types), events=count)

    @classmethod
    def from_small_controls(cls, path: str | Path, *, tensor_clock_ghz: float):
        """Fit a compound active-union tail, not isolated instruction latency.

        Subtract the documented MM initiation demand, then fit nonnegative
        kernel startup plus repeat-count tail to median independent controls.
        There is no target-operator or child timing input. Partial/control to
        other geometry and source sequence transport remains unvalidated.
        """
        import numpy as np
        from scipy.optimize import nnls
        grouped = {}
        with Path(path).open(newline="") as file:
            for row in csv.DictReader(file):
                if (row.get("kind") != "tensor_matmul_small" or row.get("status") != "ok"
                        or row.get("row_type") != "benchmark"):
                    continue
                key = (row["spec.dtype"], *(int(row[f"spec.{k}"]) for k in ("m", "k", "n", "repeat")))
                if key[1:3] != (64, 64):
                    raise ValueError("unexpected independent control geometry")
                grouped.setdefault(key, []).append(float(row["profile.tensor_engine_active_time"])*1e9)
        points, diagnostics = {}, {}
        for dtype in sorted({key[0] for key in grouped}):
            samples = [(key, statistics.median(values)) for key,values in grouped.items() if key[0]==dtype]
            if len(samples) < 6:
                raise ValueError("insufficient width/repeat controls")
            precision = 4. if dtype == "float32" else 1.
            x = np.asarray([[1., key[4]] for key,y in samples])
            ii = np.asarray([precision*max(key[3],64)/tensor_clock_ghz*key[4] for key,y in samples])
            y = np.asarray([y for key,y in samples])
            coefficients = nnls(x, y-ii)[0]
            points[dtype] = tuple(map(float, coefficients))
            predicted = x@coefficients+ii
            diagnostics[dtype] = dict(samples=len(samples),
                training_wape=float(np.abs(predicted-y).sum()/y.sum()),
                target_labels_used=False, completion_tail_is_identified_service=False)
        if not points:
            raise ValueError("no valid independent small controls")
        return cls(tensor_clock_ghz, points), diagnostics
