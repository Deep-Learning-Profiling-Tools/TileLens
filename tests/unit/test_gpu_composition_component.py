import pytest

from microbench.gpu.common.cases import load_cases


def test_component_matrix_matches_second_dot_without_new_fold_leakage():
    cases = load_cases("composition_component", "control")
    parents = [
        c
        for c in load_cases("resource_transfer", "control")
        if c["kind"] == "structure_composition" and c["variant"] == 2
    ]
    assert len(cases) == len(parents) == 32
    assert len({c["id"] for c in cases}) == 32
    assert not load_cases("composition_component", "holdout")
    for component in cases:
        parent = next(
            p
            for p in parents
            if all(
                p[k] == component[k]
                for k in (
                    "bm",
                    "bn",
                    "dtype",
                    "precision",
                    "num_warps",
                    "num_stages",
                    "programs",
                )
            )
        )
        assert component["bk"] == parent["bn"]
        assert component["cv_group"] == parent["cv_group"]


@pytest.mark.parametrize(
    "dtype,precision",
    [
        ("float32", "ieee"),
        ("float32", "tf32"),
        ("bfloat16", "ieee"),
        ("float16", "ieee"),
    ],
)
def test_component_source_only_numerical(dtype, precision, monkeypatch):
    import torch
    import triton
    import triton_viz
    from microbench.gpu.tests.coverage.kernels import prepare, check_output
    from triton_viz.performance.triton_observe import observe

    def forbidden(*args, **kwargs):
        raise AssertionError("No CUDA or compilation in source control validation")

    monkeypatch.setattr(triton, "compile", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    case = next(
        c
        for c in load_cases("composition_component", "control")
        if c["dtype"] == dtype and c["precision"] == precision
    )
    try:
        kernel, grid, inputs, output = prepare(case, "cpu")
        source = observe(
            kernel,
            grid,
            *inputs,
            num_warps=case["num_warps"],
            num_stages=case["num_stages"],
        )
        check_output(case, output)
        assert source["program_count"] == 48
    finally:
        triton_viz.clear()
