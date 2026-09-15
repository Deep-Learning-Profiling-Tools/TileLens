import pytest

from microbench.gpu.common.cases import load_cases
from microbench.gpu.tests.coverage.kernels import prepare, check_output
from triton_viz.performance.triton_observe import observe
from triton_viz.tools.gpu_distribution_experiments import enrich


@pytest.mark.parametrize(
    "case", load_cases("coverage", "holdout"), ids=lambda c: c["id"]
)
def test_fresh_validation_source(case, monkeypatch):
    import triton
    import triton_viz
    import torch

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU validation cannot compile or initialize CUDA")

    monkeypatch.setattr(triton.compiler, "compile", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    try:
        kernel, grid, args, output = prepare(case, "cpu")
        source = observe(kernel, grid, *args)
        check_output(case, output)
        enrich(source, 48)
    finally:
        triton_viz.clear()


def test_fresh_validation_is_disjoint():
    controls = load_cases("coverage", "control")
    holdouts = load_cases("coverage", "holdout")
    assert len(holdouts) == 32
    for field in ("id", "kind", "cv_group"):
        assert {c[field] for c in controls}.isdisjoint(c[field] for c in holdouts)
