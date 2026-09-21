"""Observe Triton through CPU interpretation, without target compilation.

The trace is concrete: use representative inputs for data-dependent branches.
Every launched program is observed; there is no implicit grid extrapolation.
"""

from __future__ import annotations

import math
from contextlib import contextmanager, ExitStack
from typing import Any

import numpy as np

from triton_viz.core.callbacks import ForLoopCallbacks, OpCallbacks
from triton_viz.core.client import Client


class PerformanceTrace(Client):
    NAME = "performance_trace"

    def __init__(self, *, capture_cache=False, capture_loops=False):
        super().__init__()
        self.events: list[dict[str, Any]] = []
        self.grid: tuple[int, ...] = ()
        self._values: dict[int, int] = {}
        self._keepalive: list[Any] = []
        self._load_sectors: set[int] = set()
        self._load_requests = 0
        self._store_requests = 0
        self._program_loads: dict[tuple[int, ...], int] = {}
        self.capture_cache = capture_cache
        self._cache_ids: dict[int, int] = {}
        self._cache_accesses: list[dict[str, Any]] = []
        self.capture_loops = capture_loops
        self._loop_sites = {}
        self._loop_kinds = {}
        self._loop_stacks = {}
        self._loops = []

    def pre_run_callback(self, fn):
        return True

    def post_run_callback(self, fn):
        return True

    def pre_warmup_callback(self, jit_fn, *args, **kwargs):
        return False

    def post_warmup_callback(self, jit_fn, ret):
        pass

    def arg_callback(self, name, arg, arg_cvt):
        pass

    def grid_callback(self, grid):
        self.grid = tuple(int(v) for v in grid)

    def grid_idx_callback(self, grid_idx):
        self.grid_idx = tuple(int(v) for v in grid_idx)

    def register_for_loop_callback(self):
        if not self.capture_loops:
            return ForLoopCallbacks()

        def range_type(site, kind):
            self._loop_sites.setdefault(site, len(self._loop_sites))
            self._loop_kinds[site] = kind

        def before(site, iterable):
            program = tuple(self.grid_idx or ())
            stack = self._loop_stacks.setdefault(program, [])
            row = dict(
                site=self._loop_sites[site],
                kind=self._loop_kinds[site],
                program=list(program),
                depth=len(stack),
                event_start=len(self.events),
                event_end=None,
                iterations=[],
                complete=False,
            )
            self._loops.append(row)
            stack.append((site, row))

        def iteration(site, index):
            stack = self._loop_stacks[tuple(self.grid_idx or ())]
            if not stack or stack[-1][0] != site:
                raise ValueError("Incomplete nested loop observation")
            row = stack[-1][1]
            if row["iterations"]:
                row["iterations"][-1]["event_end"] = len(self.events)
            row["iterations"].append(dict(event_start=len(self.events), event_end=None))

        def after(site):
            stack = self._loop_stacks[tuple(self.grid_idx or ())]
            if not stack or stack[-1][0] != site:
                raise ValueError("Incomplete nested loop observation")
            _, row = stack.pop()
            row["event_end"] = len(self.events)
            row["complete"] = True
            if row["iterations"]:
                row["iterations"][-1]["event_end"] = len(self.events)

        return ForLoopCallbacks(
            range_type_callback=range_type,
            before_loop_callback=before,
            loop_iter_listener=iteration,
            after_loop_callback=after,
        )

    def finalize(self):
        self._keepalive.clear()
        return self.events

    @staticmethod
    def _array(value):
        value = getattr(value, "handle", value)
        data = getattr(value, "data", None)
        return data if isinstance(data, np.ndarray) else None

    def register_op_callback(self, op_type):
        name = op_type.name

        def before(*args, **kwargs):
            stack = self._get_thread_local("raw_memory_stack", [])
            stack.append(False)
            self._set_thread_local("raw_memory_stack", stack)

        @self.lock_fn
        def after(ret, *args, **kwargs):
            stack = self._get_thread_local("raw_memory_stack", [])
            if name in {"raw_load", "raw_store"}:
                # Triton's raw operations delegate to masked operations in
                # some versions. Count the underlying transfer exactly once,
                # while still supporting versions with standalone raw ops.
                if stack.pop():
                    return
            elif name in {"load", "store"} and stack:
                stack[-1] = True
            output = self._array(ret)
            arrays = [(arg, self._array(arg)) for arg in args]
            arrays = [(arg, arr) for arg, arr in arrays if arr is not None]
            seq = len(self.events)
            deps = sorted(
                {self._values[id(arr)] for _, arr in arrays if id(arr) in self._values}
            )
            event = {
                "seq": seq,
                "op": name,
                "program": list(self.grid_idx or ()),
                "dependencies": deps,
                "shape": list(output.shape) if output is not None else [],
                "dtype": str(
                    getattr(ret, "dtype", output.dtype if output is not None else "")
                ),
                "input_shapes": [list(arr.shape) for _, arr in arrays],
                "elements": int(output.size) if output is not None else 0,
            }
            if name in {"binary_op", "unary_op"}:
                function = next((arg for arg in args if callable(arg)), None)
                event["primitive"] = getattr(function, "__name__", "unknown")
            if name in {"load", "raw_load", "store", "raw_store"}:
                ptr = self._array(args[0])
                # Memory width comes from the pointer, including masked stores
                # whose frontend adapter does not expose the stored value.
                mask = (
                    self._array(args[1])
                    if name in {"load", "store"} and len(args) > 1
                    else None
                )
                active = (
                    np.ones(ptr.shape, dtype=bool)
                    if mask is None
                    else np.broadcast_to(mask, ptr.shape)
                )
                dtype = getattr(getattr(args[0], "dtype", None), "element_ty", None)
                # BF16 interpreter storage may be float32; semantic width wins.
                width = getattr(dtype, "primitive_bitwidth", None)
                if not width:
                    raise ValueError("Cannot determine semantic memory element width")
                item_bytes = max(1, int(width // 8))
                addresses = ptr[active].astype(np.uint64)
                sector_ids = np.unique(addresses // 32)
                if item_bytes > 1 and addresses.size:
                    sector_ids = np.unique(
                        np.concatenate(
                            (addresses // 32, (addresses + item_bytes - 1) // 32)
                        )
                    )
                sectors = int(sector_ids.size)
                if self.capture_cache:
                    blocks = []
                    for sector in map(int, sector_ids):
                        if sector not in self._cache_ids:
                            self._cache_ids[sector] = len(self._cache_ids)
                        blocks.append(self._cache_ids[sector])
                    self._cache_accesses.append(
                        dict(
                            seq=seq,
                            program=list(self.grid_idx or ()),
                            op="load" if name in {"load", "raw_load"} else "store",
                            blocks=blocks,
                        )
                    )
                if name in {"load", "raw_load"}:
                    self._load_sectors.update(map(int, sector_ids))
                    self._load_requests += sectors
                    program = tuple(self.grid_idx or ())
                    self._program_loads[program] = (
                        self._program_loads.get(program, 0) + sectors
                    )
                else:
                    self._store_requests += sectors
                event.update(
                    bytes=int(active.sum()) * item_bytes,
                    sectors=int(sectors),
                    item_bytes=item_bytes,
                    elements=int(ptr.size),
                    masked=not bool(np.all(active)),
                )
            self.events.append(event)
            if output is not None:
                self._values[id(output)] = seq
                self._keepalive.append(output)

        @self.lock_fn
        def raw_after(ret, *args, **kwargs):
            # Adapters omit store values and dot accumulators. Recover these
            # edges without changing existing clients' adapted hook contracts.
            # A delegated raw transfer has already been recorded by its masked
            # operation, including its operands, so do not record it twice.
            if not self.events or self.events[-1]["op"] != name:
                return
            event = self.events[-1]
            if name == "dot":

                def operand(index, key):
                    return args[index] if len(args) > index else kwargs.get(key)

                event["dot_input_dtypes"] = [
                    str(getattr(operand(i, key), "dtype", "unknown"))
                    for i, key in enumerate(("a", "b"))
                ]
                event["dot_accumulator_dtype"] = str(
                    getattr(operand(2, "d"), "dtype", "unknown")
                )
                precision = operand(3, "input_precision")
                event["dot_input_precision"] = (
                    str(getattr(precision, "name", precision))
                    .lower()
                    .rsplit(".", 1)[-1]
                    if precision is not None
                    else "unknown"
                )
            dependencies = set(event["dependencies"])
            for value in (*args, *kwargs.values()):
                array = self._array(value)
                dependency = self._values.get(id(array)) if array is not None else None
                if dependency is not None and dependency < event["seq"]:
                    dependencies.add(dependency)
            event["dependencies"] = sorted(dependencies)

        return OpCallbacks(
            before_callback=before if name in {"raw_load", "raw_store"} else None,
            after_callback=after,
            raw_after_callback=raw_after
            if name in {"store", "raw_store", "dot"}
            else None,
        )


@contextmanager
def _bf16_constant_compat():
    """Fix BF16 scalar construction and uint16 arithmetic in CPU interpretation."""
    from unittest.mock import patch

    import triton.language as tl
    from triton.runtime.interpreter import InterpreterBuilder, TensorHandle

    def encode(values):
        values = np.asarray(values, dtype=np.float32)
        bits = values.view(np.uint32)
        # Round to nearest, ties to even; preserve NaN rather than rounding it
        # into infinity. The interpreter's generic converter rounds ties up.
        rounded = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(
            np.uint16
        )
        rounded[np.isnan(values)] = np.uint16(0x7FC0)
        return TensorHandle(rounded, tl.bfloat16)

    def get_bf16(self, value):
        return encode([value])

    def decode(value):
        if value.dtype.scalar == tl.bfloat16:
            return TensorHandle(
                (value.data.astype(np.uint32) << 16).view(np.float32), tl.float32
            )
        return value

    from triton_viz.core.frontend.base import get_frontend

    frontend = get_frontend("triton")
    operations = frontend.original_ops[frontend.builder]
    original_binary, original_dot = operations["binary_op"], operations["create_dot"]

    def binary(lhs, rhs, op):
        result = original_binary(decode(lhs), decode(rhs), op)
        return encode(result.data) if lhs.dtype.scalar == tl.bfloat16 else result

    def dot(a, b, *args, **kwargs):
        return original_dot(decode(a), decode(b), *args, **kwargs)

    with ExitStack() as stack:
        if not hasattr(InterpreterBuilder, "get_bf16"):
            stack.enter_context(
                patch.object(InterpreterBuilder, "get_bf16", get_bf16, create=True)
            )
        stack.enter_context(
            patch.dict(operations, {"binary_op": binary, "create_dot": dot})
        )
        yield


def observe(
    kernel,
    grid,
    *args,
    num_warps=4,
    num_stages=2,
    capture_cache=False,
    capture_loops=False,
    **kwargs,
):
    """Interpret a fixed Triton launch on CPU and return portable source facts.

    Pass CPU tensors. Autotuners must be resolved to a fixed configuration first.
    Source interpretation executes stores into those CPU tensors.
    Optional cache accesses use canonical sector IDs, including stores/aliases.
    Their order is the interpreter's order, not a measured hardware schedule.
    """
    import triton_viz

    if any(getattr(arg, "is_cuda", False) for arg in (*args, *kwargs.values())):
        raise ValueError("Observe requires CPU inputs, independent of target execution")
    if hasattr(kernel, "configs"):
        raise ValueError("Resolve autotuning before pre-compile prediction")
    trace = PerformanceTrace(capture_cache=capture_cache, capture_loops=capture_loops)
    with _bf16_constant_compat():
        triton_viz.trace(trace)(kernel)[grid](
            *args, num_warps=num_warps, num_stages=num_stages, **kwargs
        )
    source = {
        "schema": "triton-viz.gpu-source.v1",
        "grid": list(trace.grid),
        "program_count": math.prod(trace.grid),
        "num_warps": num_warps,
        "num_stages": num_stages,
        "events": trace.events,
        # Addresses exist only transiently during observation. Export counts,
        # never process-specific pointers. Masking/raw delegation use the same
        # audited access path as legacy per-event sector accounting.
        "memory_working_set": {
            "schema": "triton-viz.gpu-memory-working-set.v1",
            "load_unique_sectors": len(trace._load_sectors),
            "load_sector_requests": trace._load_requests,
            "store_sector_requests": trace._store_requests,
            "program_load_sectors_p90": float(
                np.quantile(
                    list(trace._program_loads.values())
                    + [0] * (math.prod(trace.grid) - len(trace._program_loads)),
                    0.9,
                )
            ),
        },
    }
    if capture_cache:
        source["cache_access_trace"] = dict(
            schema="triton-viz.gpu-source-cache-access.v1",
            block_bytes=32,
            order="interpreter_program_order_not_hardware_schedule",
            unique_blocks=len(trace._cache_ids),
            accesses=trace._cache_accesses,
        )
    if capture_loops:
        source["loop_trace"] = dict(
            schema="triton-viz.gpu-source-loops.v1",
            loops=trace._loops,
            complete=all(row["complete"] for row in trace._loops),
        )
    return source
