import copy

import pytest

from microbench.gpu.common.cases import load_cases
from triton_viz.tools.gpu_resource_transfer_audit import FEATURES, audit, train, predict


def test_k_transfer_matrix_preserves_all_original_cross_factors():
    from triton_viz.tools.gpu_control_resources import selected_controls

    original = load_cases("pressure", "control")
    rows = selected_controls("pressure_k_transfer")
    assert len(rows) == len({r["id"] for r in rows}) == 128
    assert load_cases("pressure_k_transfer", "holdout") == []
    for case in original:
        children = [r for r in rows if r["id"].startswith(case["id"] + "_k")]
        assert {(r["bk"], r["num_stages"]) for r in children} == {
            (16, 1),
            (16, 2),
            (64, 1),
            (64, 2),
        }
        for child in children:
            assert {
                k: v
                for k, v in child.items()
                if k not in {"id", "bk", "num_stages", "cv_group"}
            } == {
                k: v
                for k, v in case.items()
                if k not in {"id", "bk", "num_stages", "cv_group"}
            }
            assert (
                child["cv_group"]
                == f"pressure_{child['bm']}x{child['bn']}x{child['bk']}"
            )


def test_layout_descriptors_ignore_labels_and_mark_tensor_gap():
    from triton_viz.tools.gpu_resource_transfer_audit import layout_features

    row = dict(
        source_features=dict(threads_per_program=128),
        source_liveness={},
        program_count=48,
        operation_counts=dict(dot=48),
        dot_shapes=[[[64, 64], [64, 64]]],
        source_precision=[["fp32", "fp32", "ieee"]],
        local_bytes_per_thread=object(),
        registers_per_thread=object(),
        median_us=object(),
    )
    features = layout_features(row, compiler_version="3.7.0")
    assert features["max_simt_operand_fragment_words"] == 768
    assert features["simt_layout_available"] == 1
    row["source_precision"] = [["fp16", "fp16", "ieee"]]
    features = layout_features(row, compiler_version="3.7.0")
    assert features["simt_layout_available"] == 0
    assert features["max_simt_operand_fragment_words"] == 0
    with pytest.raises(ValueError, match="version"):
        layout_features(row, compiler_version="unknown")


@pytest.mark.parametrize(
    "dtype,precision,lo,hi",
    [("fp16", "ieee", 96, 96), ("bf16", "ieee", 96, 96), ("fp32", "tf32", 192, 192)],
)
def test_mma_materialization_does_not_read_compiler_labels_or_dynamic_ancestry(
    dtype, precision, lo, hi
):
    from triton_viz.tools.gpu_resource_transfer_audit import (
        mma_materialization_features,
    )

    row = dict(
        source_features=dict(threads_per_program=128),
        source_liveness={},
        program_count=48,
        operation_counts=dict(dot=240),
        dot_shapes=[[[256, 32], [32, 128]]],
        source_precision=[[dtype, dtype, precision]],
        case=object(),
        dot_ancestry=object(),
        artifacts=object(),
        registers_per_thread=object(),
        local_bytes_per_thread=object(),
        median_us=object(),
    )
    result = mma_materialization_features(row, compiler_version="3.7.0")
    assert result["mma_materialization_available"] == 1
    assert result["min_mma_operand_fragment_words"] == lo
    assert result["max_mma_operand_fragment_words"] == hi
    assert result["min_mma_accumulator_fragment_words"] == 256
    assert result["max_mma_accumulator_fragment_words"] == 256
    row["dot_shapes"] = [[[16, 32], [32, 8]]]
    assert (
        mma_materialization_features(row, compiler_version="3.7.0")[
            "mma_materialization_available"
        ]
        == 0
    )
    with pytest.raises(ValueError, match="version"):
        mma_materialization_features(row, compiler_version="unknown")


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


@pytest.mark.parametrize("feature_set", ["initial_layout", "mma_materialization"])
def test_layout_fold_prediction_cannot_read_its_validation_allocation(feature_set):
    data = report()
    data["compiler_version"] = "3.7.0"
    for n, row in enumerate(data["rows"]):
        row.update(
            program_count=1,
            operation_counts=dict(dot=1),
            source_liveness=dict(logical_live_float_words_per_thread=64),
            dot_shapes=[[[32 * 2**n, 32], [32, 64]]],
            source_precision=[["fp32", "fp32", "ieee"]],
        )
        row["source_features"]["threads_per_program"] = 128
    first = audit(data, feature_set=feature_set)
    data["rows"][0]["local_bytes_per_thread"] = 1000000
    data["rows"][0]["median_us"] = object()
    second = audit(data, feature_set=feature_set)
    assert first["rows"][0]["prediction"] == second["rows"][0]["prediction"]
    assert "case0" not in first["rows"][0]["training_ids"]


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
