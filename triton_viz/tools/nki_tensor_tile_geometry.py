"""Experimental geometry-dependent Tensor response, rather than FLOP sharing.

The reference unit is a normalized Dot with M=128, K=128, N=512. Coefficients
may be initialized from homogeneous reference-tile controls. Expanding them to
other shapes is an explicit modeling hypothesis, not a compiler instruction
count, a pure service price, or a proven performance bound. Source view reuse
is also a feature hypothesis; physical stationary-weight reuse is not inferred.
"""
from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any


def _dtype(value: str) -> str:
    value = str(value).lower()
    return {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}.get(value, value)


@dataclass(frozen=True)
class TensorTileGeometryCalibration:
    """Affine empirical response in reference-equivalent tile/view units.

    Tuple entries are startup, Dot, stationary-view, moving-view, output-view
    nanoseconds. Only startup is once per dtype per kernel. View terms are
    amortized over uses of the same explicitly identified source view. Missing
    view identity is occurrence-specific and never interpreted as free reuse.
    """

    points: dict[str, tuple[float, float, float, float, float]]
    reference_m: int = 128
    reference_k: int = 128
    reference_n: int = 512

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in (
            self.reference_m, self.reference_k, self.reference_n
        )):
            raise ValueError("positive reference tile dimensions required")
        if not self.points:
            raise ValueError("nonempty Tensor calibration required")
        for dtype, values in self.points.items():
            if dtype != _dtype(dtype) or len(values) != 5:
                raise ValueError("canonical dtype and five response coefficients required")
            if any(not math.isfinite(v) or v < 0 for v in values) or values[1] <= 0:
                raise ValueError("finite nonnegative coefficients and positive Dot response required")

    @classmethod
    def from_source_geometry_csv(cls, path: str | Path):
        """Initialize from reference-tile controls; does not validate transport."""
        points = {}
        with Path(path).open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row.get("geometry_model") or row.get("revisit_ns"):
                    raise ValueError("revisit surfaces need a separate geometry response fit")
                dtype = _dtype(row["dtype"])
                if dtype in points:
                    raise ValueError("duplicate Tensor dtype calibration")
                points[dtype] = tuple(float(row.get(k) or 0) for k in (
                    "startup_ns", "dot_ns", "lhs_tile_ns", "rhs_tile_ns", "output_tile_ns"
                ))
        return cls(points)

    def geometry(self, event: dict[str, Any]) -> dict[str, Any]:
        a, b = event.get("input_shape"), event.get("other_shape")
        if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)) or len(a) != 2 or len(b) != 2:
            raise ValueError("normalized 2-D Dot shapes required for Tensor geometry model")
        if any(type(v) is not int or v <= 0 for v in [*a, *b]) or a[1] != b[0]:
            raise ValueError("positive compatible normalized Dot shapes required")
        m, k, n = a[0], a[1], b[1]
        types = [_dtype(v) for v in event.get("input_dtypes", ()) if _dtype(v) not in {"bool", "boolean"}]
        if len(set(types)) != 1:
            raise ValueError("one explicit Tensor operand dtype required")
        dtype = types[0]
        if dtype not in self.points:
            raise ValueError(f"missing Tensor geometry response for {dtype}")
        mt, kt, nt = (math.ceil(m/self.reference_m), math.ceil(k/self.reference_k), math.ceil(n/self.reference_n))
        return dict(dtype=dtype, shape=[m,k,n], dot_units=mt*kt*nt,
                    stationary_view_units=mt*kt, moving_view_units=kt*nt,
                    output_view_units=mt*nt,
                    reference_shape=[self.reference_m,self.reference_k,self.reference_n],
                    nonreference_shape=(m,k,n)!=(self.reference_m,self.reference_k,self.reference_n))

    @staticmethod
    def _view(event: dict[str, Any], role: str, occurrence: int):
        if role == "output":
            storage, bounds = event.get("output_storage"), event.get("output_range")
        else:
            index = 0 if role == "stationary" else 1
            storages, ranges = event.get("input_storages") or [], event.get("input_ranges") or []
            storage = storages[index] if index < len(storages) else None
            bounds = ranges[index] if index < len(ranges) else None
        if (not isinstance(storage, int) or not isinstance(bounds, (list, tuple))
                or len(bounds) != 2 or any(not isinstance(v, (int, float))
                or not math.isfinite(v) for v in bounds) or bounds[1] <= bounds[0]):
            return (role,"unresolved_occurrence",occurrence)
        # Include shape: equal bounding intervals do not establish equal views.
        shape = (event.get("output_shape") if role=="output" else
                 event.get("input_shape") if role=="stationary" else event.get("other_shape"))
        versions = event.get("input_versions") or []
        version = versions[index] if role != "output" and index < len(versions) else None
        return role, storage, tuple(bounds), tuple(shape or ()), tuple(event.get("input_dtypes") or ()), version

    def assign_events(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        dots = [e for e in events if e.get("op")=="dot"]
        geometry = [self.geometry(e) for e in dots]
        keys = [{role:self._view(e,role,i) for role in ("stationary","moving","output")} for i,e in enumerate(dots)]
        uses = Counter(key for row in keys for key in row.values())
        work = 0.0
        for e,g,k in zip(dots,geometry,keys):
            _,dot_ns,lhs_ns,rhs_ns,out_ns = self.points[g["dtype"]]
            value = (dot_ns*g["dot_units"]
                + lhs_ns*g["stationary_view_units"]/uses[k["stationary"]]
                + rhs_ns*g["moving_view_units"]/uses[k["moving"]]
                + out_ns*g["output_view_units"]/uses[k["output"]])
            e["tensor_tile_geometry_work_ns"] = value
            # Supersede any stale aggregate Tensor override from an earlier simulation.
            e["scheduler_duration_override_ns"] = value
            e["tensor_tile_geometry"] = g
            work += value
        startup = sum(self.points[d][0] for d in {g["dtype"] for g in geometry})
        return dict(startup_ns=startup,work_ns=work,dot_events=len(dots),
                    dot_units=sum(g["dot_units"] for g in geometry),
                    nonreference_dot_events=sum(g["nonreference_shape"] for g in geometry),
                    geometry_transport_validated=False,physical_instruction_count_established=False,
                    source_view_reuse_is_physical_reuse=False)

    def isolated_work_ns(self, event: dict[str, Any]) -> float:
        probe = dict(event)
        self.assign_events([probe])
        return probe["tensor_tile_geometry_work_ns"]
