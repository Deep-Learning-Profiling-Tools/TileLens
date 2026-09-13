import pytest

from microbench.gpu.tests.compositional.kernels import (
    cases,
    prepare,
    check_output,
)
from triton_viz.performance.triton_observe import observe
from triton_viz.tools.gpu_distribution_experiments import enrich


@pytest.mark.parametrize(
    "case", cases("control")[36:42] + cases("holdout")[:4], ids=lambda c: c["id"]
)
def test_compositional_source_outputs_and_work(case):
    import triton_viz

    kernel, grid, args, out = prepare(case, "cpu")
    source = observe(kernel, grid, *args)
    check_output(case, out)
    features, _, _ = enrich(source, 48)
    if case["kind"] == "reduction_chain":
        assert features["program_reduction_p90"] == 9 * case["repeat"]
        assert features["path_reduction_p90"] == 9 * case["repeat"]
    assert sum(e.get("bytes", 0) for e in source["events"]) == 8 * case["n"]
    triton_viz.clear()
