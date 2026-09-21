from pathlib import Path

import pytest

from microbench.gpu.common.cases import load_cases
from microbench.gpu.tests.coverage.kernels import prepare, check_output
from triton_viz.performance.gpu import expand
from triton_viz.performance.gpu_memory import memory_features
from triton_viz.performance.triton_observe import observe


def test_structure_audit_retains_all_contrasts_and_negative_increments(
    tmp_path, monkeypatch
):
    import json
    from triton_viz.tools.gpu_dot_structure_audit import audit

    cases = load_cases("structure", "control")
    (tmp_path / "controls").mkdir()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"fingerprint": "test", "splits": {"control": cases, "holdout": []}})
    )
    for case in cases:
        (tmp_path / "controls" / (case["id"] + ".json")).write_text(
            json.dumps(
                {
                    "role": "control",
                    "case": case,
                    "cv_group": case["cv_group"],
                    "contaminated": False,
                    "fingerprint": "test",
                    "latency_us": 9 if case["variant"] in {"strided", 1} else 10,
                }
            )
        )
    original = Path.read_text

    def read(path, *args, **kwargs):
        assert (
            path.name in {"manifest.json", "structure_control.json"}
            or path.parent == tmp_path / "controls"
        )
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    report = audit(tmp_path)
    assert report["n"] == 96 and len(report["contrasts"]) == 40
    for contrast in report["contrasts"]:
        if "strided_over_packed" in contrast:
            assert contrast["strided_over_packed"] == 0.9
        else:
            assert contrast["normalization_increment_us"] == -1
    (tmp_path / "controls" / (cases[0]["id"] + ".json")).unlink()
    with pytest.raises(FileNotFoundError):
        audit(tmp_path)


def test_structure_controls_are_fixed_and_do_not_read_holdouts(monkeypatch):
    original = Path.read_text

    def read(path, *args, **kwargs):
        assert path.name == "structure_control.json"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    cases = load_cases("structure", "control")
    assert load_cases("structure", "holdout") == []
    assert len(cases) == len({c["id"] for c in cases}) == 96
    assert sum(c["kind"] == "structure_stream" for c in cases) == 48
    assert {c["cv_group"] for c in cases} == {"8", "64"}


@pytest.mark.parametrize(
    "dtype,precision",
    [
        ("float32", "ieee"),
        ("float32", "tf32"),
        ("bfloat16", "ieee"),
        ("float16", "ieee"),
    ],
)
def test_stream_layout_pairs_and_composition_stages_on_cpu(
    dtype, precision, monkeypatch
):
    import torch
    import triton
    import triton_viz

    def forbidden(*args, **kwargs):
        raise AssertionError("Control source audit must precede target compilation")

    monkeypatch.setattr(triton.compiler, "compile", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    results = []
    dot_counts = []
    try:
        for family, variant in [
            ("stream", "packed"),
            ("stream", "strided"),
            ("composition", 0),
            ("composition", 1),
            ("composition", 2),
        ]:
            case = dict(
                kind=f"structure_{family}",
                variant=variant,
                programs=4,
                bm=32,
                bn=64,
                bk=32,
                repeat=3 if family == "stream" else 1,
                dtype=dtype,
                precision=precision,
            )
            kernel, grid, args, out = prepare(case, "cpu")
            source = observe(kernel, grid, *args, num_warps=8, num_stages=2)
            check_output(case, out)
            assert not expand(source, sm_count=48)["ood_reasons"]
            features, reasons = memory_features(source)
            assert not reasons
            if family == "stream":
                results.append((out.clone(), features))
            else:
                dot_counts.append(sum(e["op"] == "dot" for e in source["events"]))
            triton_viz.clear()
        torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
        assert results[0][1] == results[1][1]
        assert dot_counts == [4, 4, 8]
    finally:
        triton_viz.clear()
