import pytest

from microbench.gpu.common.cases import load_cases


def test_pressure_matrix_keeps_all_declared_combinations():
    cases = load_cases("pressure", "control")
    assert len(cases) == len({c["id"] for c in cases}) == 32
    assert load_cases("pressure", "holdout") == []
    assert {c["programs"] for c in cases} == {48}
    assert {c["num_warps"] for c in cases} == {4, 8}
    assert {c["repeat"] for c in cases} == {5}
    assert {c["reuse"] for c in cases} == {"none"}
    assert len({c["cv_group"] for c in cases}) == 4


@pytest.mark.parametrize(
    "dtype,precision",
    [
        ("float32", "ieee"),
        ("float32", "tf32"),
        ("bfloat16", "ieee"),
        ("float16", "ieee"),
    ],
)
def test_largest_pressure_tile_source_before_compile(dtype, precision, monkeypatch):
    import torch
    import triton
    import triton_viz
    from microbench.gpu.tests.coverage.geometry import prepare, check_output
    from triton_viz.performance.triton_observe import observe

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU pressure audit must not compile or initialize CUDA")

    monkeypatch.setattr(triton.compiler, "compile", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    case = next(
        c
        for c in load_cases("pressure", "control")
        if c["bm"] == c["bn"] == 256
        and c["dtype"] == dtype
        and c["precision"] == precision
    )
    # Single-program unit test of the exact tile/loop; declaration stays at 48.
    case = {**case, "programs": 1}
    kernel, grid, inputs, out = prepare(case, "cpu")
    try:
        source = observe(
            kernel,
            grid,
            *inputs,
            num_warps=case["num_warps"],
            num_stages=case["num_stages"],
        )
        check_output(case, out)
        assert sum(e["op"] == "dot" for e in source["events"]) == 5
    finally:
        triton_viz.clear()
