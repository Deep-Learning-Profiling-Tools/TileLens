from collections import Counter
from pathlib import Path

import pytest

from microbench.gpu.common.cases import load_cases


def test_stability_controls_balance_families_without_changing_templates(monkeypatch):
    original = Path.read_text

    def guarded(path, *args, **kwargs):
        assert "holdout" not in path.name
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    templates = {
        c["id"]: c
        for suite in ("precision", "geometry", "structure")
        for c in load_cases(suite, "control")
        if c.get("programs") == 8
    }
    cases = load_cases("stability", "control")
    assert load_cases("stability", "holdout") == []
    assert len(templates) == 232
    assert len(cases) == len({c["id"] for c in cases}) == 696
    for programs in (48, 96, 384):
        subset = [c for c in cases if c["programs"] == programs]
        assert Counter(c["kind"] for c in subset) == {
            "coverage_vector": 88,
            "coverage_dot": 24,
            "geometry_dot": 72,
            "structure_stream": 24,
            "structure_composition": 24,
        }
        assert {c["template_id"] for c in subset} == set(templates)
        assert {c["cv_group"] for c in subset} == {str(programs)}
        for case in subset:
            original_case = templates[case["template_id"]]
            for key, value in original_case.items():
                if key not in {"id", "programs", "cv_group", "pair_id"}:
                    assert case[key] == value
            if "pair_id" in case:
                assert (
                    case["pair_id"]
                    == f"stability_p{programs}__{original_case['pair_id']}"
                )


@pytest.mark.parametrize("programs,waves", [(48, 1), (96, 2), (384, 8)])
def test_stability_launches_cover_new_waves_before_compilation(
    programs, waves, monkeypatch
):
    import torch
    import triton
    import triton_viz
    from microbench.gpu.tests.coverage.kernels import prepare, check_output
    from triton_viz.performance.gpu import expand
    from triton_viz.performance.triton_observe import observe

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU audit must not compile or initialize CUDA")

    monkeypatch.setattr(triton.compiler, "compile", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    case = next(
        c
        for c in load_cases("stability", "control")
        if c["programs"] == programs and c["kind"] == "coverage_vector"
    )
    try:
        kernel, grid, args, out = prepare(case, "cpu")
        source = observe(
            kernel,
            grid,
            *args,
            num_warps=case["num_warps"],
            num_stages=case["num_stages"],
        )
        check_output(case, out)
        assert source["program_count"] == programs
        work = expand(source, sm_count=48)
        assert not work["ood_reasons"]
        assert work["features"]["waves"] == waves
    finally:
        triton_viz.clear()
