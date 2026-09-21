import triton
import triton.language as tl


@triton.jit
def _cache_alias(X, Y, N: tl.constexpr):
    offsets = tl.arange(0, 64)
    x = tl.load(X + offsets, offsets < N, other=0)
    tl.store(Y + offsets, x + 1, offsets < N)
    y = tl.load(Y + offsets, offsets < N, other=0)
    tl.store(Y + offsets, y + 1, offsets < N)


def test_optional_cache_source_trace_keeps_stores_aliases_and_masks(monkeypatch):
    import torch
    import triton_viz
    from triton_viz.performance.triton_observe import observe

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Source cache observation must not compile or initialize CUDA"
        )

    monkeypatch.setattr(triton.compiler, "compile", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    x, y = torch.arange(33, dtype=torch.float32), torch.empty(33)
    try:
        plain = observe(_cache_alias, (1,), x, y, 33)
        triton_viz.clear()
        captured = observe(_cache_alias, (1,), x, y, 33, capture_cache=True)
        torch.testing.assert_close(y, x + 2)
        cache = captured.pop("cache_access_trace")
        assert captured == plain
        assert cache["block_bytes"] == 32 and cache["unique_blocks"] == 10
        accesses = cache["accesses"]
        assert [a["op"] for a in accesses] == ["load", "store", "load", "store"]
        assert all(len(a["blocks"]) == 5 for a in accesses)
        assert accesses[1]["blocks"] == accesses[2]["blocks"] == accesses[3]["blocks"]
        assert set(accesses[0]["blocks"]).isdisjoint(accesses[1]["blocks"])
        assert {b for a in accesses for b in a["blocks"]} == set(range(10))
        assert all(set(a) == {"seq", "program", "op", "blocks"} for a in accesses)
    finally:
        triton_viz.clear()
