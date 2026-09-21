import copy

import pytest

from microbench.gpu.common.cases import load_cases
from triton_viz.tools.gpu_resource_transfer_audit import FEATURES, audit, train, predict


def report():
    return dict(
        role="control",
        complete=True,
        rows=[
            dict(
                case=dict(id=f"case{n}", cv_group=str(n)),
                source_features={key: n + 1 for key in FEATURES},
                source_precision=[["fp16", "fp16", "ieee"]],
                ood_reasons=[],
                registers_per_thread=10 * n,
                local_bytes_per_thread=8 * n,
            )
            for n in range(4)
        ],
    )


def test_fold_training_excludes_entire_validation_geometry():
    result = audit(report())
    assert result["count"] == 4 and not result["eligible_for_fit"]
    for row in result["rows"]:
        assert row["id"] not in row["training_ids"]
        assert set(row["neighbors"]) <= set(row["training_ids"])
    modified = report()
    modified["rows"][0]["local_bytes_per_thread"] = 100000
    modified["rows"][0]["median_us"] = object()  # Never inspect latency.
    assert audit(modified)["rows"][0]["prediction"] == result["rows"][0]["prediction"]


def test_holdout_and_incomplete_rejected():
    for field, value in [("role", "holdout"), ("complete", False)]:
        bad = report()
        bad[field] = value
        with pytest.raises(ValueError):
            audit(bad)
    with pytest.raises(ValueError):
        train([dict(role="holdout")])


def test_identical_descriptors_expose_irreducible_label_conflict():
    data = report()
    data["rows"][1]["source_features"] = dict(data["rows"][0]["source_features"])
    result = audit(data)
    assert result["descriptor_collisions"][0]["ids"] == ["case0", "case1"]
    # Absolute errors at any median sum to 8 bytes; four retained controls.
    assert result["descriptor_empirical_mae_floor"]["local_bytes_per_thread"] == 2


def test_unseen_precision_and_domain_explicit():
    rows = [dict(role="control", **row) for row in report()["rows"]]
    model = train(rows)
    frozen = copy.deepcopy(model)
    result = predict(model, {key: 100 for key in FEATURES}, [["bf16", "bf16", "ieee"]])
    assert result["prediction"] is None and "unseen_precision" in result["ood_reasons"]
    assert model == frozen


def test_resource_transfer_declares_factorial_and_no_holdout():
    cases = load_cases("resource_transfer", "control")
    assert len(cases) == len({c["id"] for c in cases}) == 192
    assert not load_cases("resource_transfer", "holdout")
    assert {c["num_warps"] for c in cases} == {4, 8}
    assert {c["num_stages"] for c in cases} == {1, 2}
    assert {c["repeat"] for c in cases} == {1, 5, 17}
    assert len({c["cv_group"] for c in cases}) == 4


@pytest.mark.parametrize("variant", [0, 1, 2])
def test_resource_compositions_cpu_numerical(variant, monkeypatch):
    import torch
    import triton
    from microbench.gpu.tests.coverage.kernels import prepare, check_output
    from triton_viz.performance.triton_observe import observe

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "CPU source observation must not compile or initialize CUDA"
        )

    monkeypatch.setattr(triton, "compile", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    case = next(
        c
        for c in load_cases("resource_transfer", "control")
        if c["kind"] == "structure_composition" and c["variant"] == variant
    )
    case = {**case, "programs": 1}
    kernel, grid, inputs, out = prepare(case, "cpu")
    observe(
        kernel,
        grid,
        *inputs,
        num_warps=case["num_warps"],
        num_stages=case["num_stages"],
    )
    check_output(case, out)


def test_source_resource_cli_forbids_compile_restores_state(tmp_path, monkeypatch):
    import json
    import torch
    import triton
    from triton_viz.tools import gpu_control_source_resources as tool

    case = next(
        c
        for c in load_cases("resource_transfer", "control")
        if c["kind"] == "structure_composition" and c["variant"] == 2
    )
    case = {**case, "programs": 1}
    monkeypatch.setattr(tool, "load_cases", lambda *_: [case])
    original = (triton.compile, torch.cuda._lazy_init, torch.get_num_threads())
    root = tmp_path / "sources"
    tool.main(["--suite", "resource_transfer", "--output", str(root)])
    assert (triton.compile, torch.cuda._lazy_init, torch.get_num_threads()) == original
    row = json.loads((root / "controls" / (case["id"] + ".json")).read_text())
    assert row["compile_and_cuda_forbidden"] and row["numerical_validation"] == "passed"
    assert row["program_count"] == 1 and row["operation_counts"]["dot"] == 2
    assert row["source_liveness"]["logical_live_float_words_per_thread"] > 0
    from triton_viz.tools.gpu_resource_transfer_audit import structural_features

    features = structural_features(row)
    assert features["dots_per_program"] == 2
    assert features["reductions_per_program"] == 2
    assert features["max_dot_k"] == 64
    assert "registers_per_thread" not in row
    with pytest.raises(ValueError, match="fresh"):
        tool.main(["--suite", "resource_transfer", "--output", str(root)])


def test_join_rejects_source_identity_and_artifact_tampering(tmp_path):
    import hashlib
    import json
    from triton_viz.tools.gpu_resource_transfer_audit import join_control_sources

    sources, compiled = tmp_path / "sources", tmp_path / "compiled"
    case = dict(id="control", cv_group="geometry")
    for root in (sources, compiled):
        (root / "controls").mkdir(parents=True)
        (root / "manifest.json").write_text(
            json.dumps(dict(role="control", cases=[case]))
        )
    source = dict(
        role="control",
        case=case,
        numerical_validation="passed",
        compile_and_cuda_forbidden=True,
    )
    source_path = sources / "controls" / "control.json"
    source_path.write_text(json.dumps(source))
    artifact = dict(
        role="control",
        case=case,
        registers_per_thread=64,
        triton_reported_spills=0,
        shared_bytes=0,
        artifacts={"ptx": "control-only"},
        artifact_sha256={"ptx": hashlib.sha256(b"control-only").hexdigest()},
    )
    artifact_path = compiled / "controls" / "control.json"
    artifact_path.write_text(json.dumps(artifact))
    assert (
        join_control_sources(sources, compiled)["rows"][0]["registers_per_thread"] == 64
    )
    source["case"] = dict(id="different")
    source_path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="mismatched"):
        join_control_sources(sources, compiled)
    artifact["artifacts"]["ptx"] = "modified"
    artifact_path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="digest"):
        join_control_sources(sources, compiled)
