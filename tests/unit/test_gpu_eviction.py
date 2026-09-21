import pytest
import torch
import triton
import triton_viz

from microbench.gpu.harness.eviction import persistent_eviction
from triton_viz.performance.triton_observe import observe


@pytest.mark.parametrize("size", [0, 1, 4095, 4096, 12289, 24577])
def test_persistent_sweep_covers_exact_declared_bytes_without_cuda(size, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU sweep validation must not compile or initialize CUDA")

    monkeypatch.setattr(triton.compiler, "compile", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    buffer = torch.full((size + 128,), 123, dtype=torch.uint8)
    try:
        observe(
            persistent_eviction,
            (3,),
            buffer[64:],
            size,
            BLOCK=4096,
            PROGRAMS=3,
            num_warps=4,
        )
        assert torch.all(buffer[:64] == 123)
        assert torch.all(buffer[64 : 64 + size] == 0)
        assert torch.all(buffer[64 + size :] == 123)
    finally:
        triton_viz.clear()
