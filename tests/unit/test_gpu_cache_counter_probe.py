import pytest

from triton_viz.tools.gpu_cache_counter_probe import (
    cache_eviction_read,
    cache_read_control,
)


@pytest.mark.parametrize("kernel", [cache_read_control, cache_eviction_read])
@pytest.mark.parametrize("dtype", ["float32", "uint8"])
def test_cache_controls_before_compilation(kernel, dtype, monkeypatch):
    import torch
    import triton
    import triton_viz
    from triton_viz.performance.triton_observe import observe

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "CPU cache-control validation must not compile or use CUDA"
        )

    monkeypatch.setattr(triton.compiler, "compile", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    x = torch.arange(130, dtype=getattr(torch, dtype))
    y = torch.empty_like(x)
    try:
        source = observe(kernel, (2,), x, y, x.numel(), 128)
        torch.testing.assert_close(y, x + 1, rtol=0, atol=0)
        assert source["program_count"] == 2
    finally:
        triton_viz.clear()
