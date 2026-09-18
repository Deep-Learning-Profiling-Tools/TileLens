from pathlib import Path

import pytest

from microbench.gpu.common.cases import load_cases
from microbench.gpu.tests.coverage.kernels import prepare, check_output
from triton_viz.performance.gpu import expand
from triton_viz.performance.gpu_dot_precision import dot_features
from triton_viz.performance.triton_observe import observe


def test_geometry_audit_never_reads_targets_and_rejects_missing_rows(
    tmp_path, monkeypatch
):
    import json
    from triton_viz.tools.gpu_dot_geometry_audit import audit

    cases = load_cases("geometry", "control")
    (tmp_path / "controls").mkdir()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"splits": {"control": cases, "holdout": []}, "fingerprint": "test"})
    )
    for case in cases:
        (tmp_path / "controls" / (case["id"] + ".json")).write_text(
            json.dumps(
                {
                    "case": case,
                    "cv_group": case["cv_group"],
                    "role": "control",
                    "fingerprint": "test",
                    "contaminated": False,
                    "features": {"tensor_flops": 1},
                    "latency_us": (10 if case["reuse"] == "none" else 5)
                    / case["num_stages"],
                }
            )
        )
    original = Path.read_text

    def guarded(path, *args, **kwargs):
        assert (
            path.name in {"geometry_control.json", "manifest.json"}
            or path.parent == tmp_path / "controls"
        )
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    report = audit(tmp_path)
    assert report["n"] == 144 and report["groups"] == 24
    assert all(p["aggregate_feature_vectors_identical"] for p in report["pairs"])
    assert set(report["median_stage2_over_stage1"].values()) == {0.5}
    assert set(report["median_reuse_over_disjoint"].values()) == {0.5}
    (tmp_path / "controls" / (cases[0]["id"] + ".json")).unlink()
    with pytest.raises(FileNotFoundError):
        audit(tmp_path)


def test_geometry_declaration_is_control_only_and_pairs_share_cv_group(monkeypatch):
    original = Path.read_text

    def guarded(path, *args, **kwargs):
        assert path.name == "geometry_control.json"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    cases = load_cases("geometry", "control")
    assert load_cases("geometry", "holdout") == []
    assert len(cases) == len({c["id"] for c in cases}) == 144
    groups = {c["cv_group"] for c in cases}
    assert len(groups) == 24
    for group in groups:
        members = [c for c in cases if c["cv_group"] == group]
        assert {(c["reuse"], c["num_stages"]) for c in members} == {
            (r, s) for r in ("none", "a", "ab") for s in (1, 2)
        }


@pytest.mark.parametrize(
    "dtype,precision",
    [
        ("float32", "ieee"),
        ("float32", "tf32"),
        ("bfloat16", "ieee"),
        ("float16", "ieee"),
    ],
)
@pytest.mark.parametrize("tile", [(64, 64, 32), (128, 128, 32), (128, 128, 64)])
def test_geometry_pairs_preserve_work_values_and_explicit_reuse(
    dtype, precision, tile, monkeypatch
):
    import torch
    import triton
    import triton_viz

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU source audit must not compile or initialize CUDA")

    monkeypatch.setattr(triton.compiler, "compile", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    bm, bn, bk = tile
    observed = []
    allocations = []
    outputs = []
    try:
        for reuse in ("none", "a", "ab"):
            case = dict(
                kind="geometry_dot",
                programs=2,
                repeat=2,
                dtype=dtype,
                precision=precision,
                bm=bm,
                bn=bn,
                bk=bk,
                reuse=reuse,
            )
            kernel, grid, args, output = prepare(case, "cpu")
            allocations.append(tuple(x.numel() for x in args[:2]))
            source = observe(kernel, grid, *args, num_warps=8, num_stages=1)
            check_output(case, output)
            features, _, reasons = dot_features(source)
            assert not reasons
            assert not expand(source, sm_count=48)["ood_reasons"]
            observed.append(features)
            outputs.append(output.clone())
            triton_viz.clear()
        assert observed[0] == observed[1] == observed[2]
        assert allocations[0][0] == 2 * allocations[1][0]
        assert allocations[0][1] == allocations[1][1] == 2 * allocations[2][1]
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        torch.testing.assert_close(outputs[0], outputs[2], rtol=0, atol=0)
    finally:
        triton_viz.clear()
