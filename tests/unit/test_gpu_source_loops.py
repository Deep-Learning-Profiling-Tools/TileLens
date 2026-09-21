import torch
import triton
import triton.language as tl

from triton_viz.performance.triton_observe import observe


@triton.jit
def nested_control(X, Y, R: tl.constexpr):
    offsets = tl.program_id(0) * 8 + tl.arange(0, 8)
    value = tl.load(X + offsets)
    for _ in range(R):
        for _ in tl.static_range(2):
            value = value + 1
    tl.store(Y + offsets, value)


@triton.jit
def break_control(X, Y):
    offset = tl.arange(0, 8)
    value = tl.load(X + offset)
    for i in range(4):
        value = value + 1
        if i == 1:
            break
    tl.store(Y + offset, value)


def test_source_loop_boundaries_are_precompile_and_preserve_default_events(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No compile/CUDA during source loop observation")

    monkeypatch.setattr(triton, "compile", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    x, y = torch.arange(16, dtype=torch.float32), torch.empty(16)
    plain = observe(nested_control, (2,), x, y, 3)
    traced = observe(nested_control, (2,), x, y, 3, capture_loops=True)
    loops = traced.pop("loop_trace")
    assert traced == plain and loops["complete"]
    torch.testing.assert_close(y, x + 6)
    assert len(loops["loops"]) == 8  # One outer plus three inner per program.
    outer = [row for row in loops["loops"] if row["depth"] == 0]
    inner = [row for row in loops["loops"] if row["depth"] == 1]
    assert len(outer) == 2 and all(len(row["iterations"]) == 3 for row in outer)
    assert len(inner) == 6 and all(len(row["iterations"]) == 2 for row in inner)
    assert all(row["event_start"] <= row["event_end"] for row in loops["loops"])


def test_zero_trip_source_loop_is_recorded():
    x, y = torch.arange(16, dtype=torch.float32), torch.empty(16)
    source = observe(nested_control, (2,), x, y, 0, capture_loops=True)
    assert source["loop_trace"]["complete"]
    assert len(source["loop_trace"]["loops"]) == 2
    assert all(row["iterations"] == [] for row in source["loop_trace"]["loops"])
    torch.testing.assert_close(y, x)


def test_early_loop_exit_is_explicitly_incomplete_not_normalized_to_trip_count():
    x, y = torch.arange(8, dtype=torch.float32), torch.empty(8)
    source = observe(break_control, (1,), x, y, capture_loops=True)
    torch.testing.assert_close(y, x + 2)
    assert not source["loop_trace"]["complete"]
    assert len(source["loop_trace"]["loops"][0]["iterations"]) == 2
