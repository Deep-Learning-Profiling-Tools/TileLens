import copy

import pytest

from microbench.gpu.common.cases import load_cases
from microbench.gpu.tests.coverage.kernels import prepare, check_output
from triton_viz.performance.gpu_dot_precision import dot_features
from triton_viz.performance.triton_observe import observe


@pytest.mark.parametrize(
    "dtype,precision,kind",
    [
        ("float32", "ieee", "ieee_fp32"),
        ("float32", "tf32", "tf32"),
        ("bfloat16", "ieee", "bf16"),
        ("float16", "ieee", "fp16"),
    ],
)
def test_observed_dot_precision_before_compilation(dtype, precision, kind, monkeypatch):
    import triton
    import triton_viz
    import torch

    def forbidden(*args, **kwargs):
        raise AssertionError("Prediction must not compile or initialize CUDA")

    monkeypatch.setattr(triton.compiler, "compile", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    case = dict(
        kind="coverage_dot",
        programs=2,
        repeat=2,
        dtype=dtype,
        precision=precision,
        bm=32,
        bn=64,
        bk=32,
    )
    try:
        kernel, grid, args, output = prepare(case, "cpu")
        source = observe(kernel, grid, *args)
        check_output(case, output)
        features, configs, reasons = dot_features(source)
        assert not reasons
        assert len(configs) == 1 and configs[0][0] == kind
        assert features[f"dot_flops_{kind}"] == 2 * 32 * 64 * 32 * 4
        assert features[f"program_dot_p90_{kind}"] == 2 * 32 * 64 * 32 * 2
        assert all(
            v == 0
            for k, v in features.items()
            if k not in {f"dot_flops_{kind}", f"program_dot_p90_{kind}"}
        )
        legacy = copy.deepcopy(source)
        for event in legacy["events"]:
            event.pop("dot_input_precision", None)
        assert dot_features(legacy)[2] == ["missing_dot_precision_metadata"]
    finally:
        triton_viz.clear()


def test_new_controls_cover_all_precisions_without_holdout_reads(monkeypatch):
    from pathlib import Path

    original = Path.read_text

    def guarded(path, *args, **kwargs):
        assert "holdout" not in path.name
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    controls = load_cases("precision", "control")
    assert len(controls) == len({c["id"] for c in controls}) == 512
    assert controls[:480] == load_cases("coverage", "control")
    assert {c["repeat"] for c in controls[480:]} == {17, 65}
    assert len({(c["dtype"], c["precision"]) for c in controls[480:]}) == 4


def test_precision_fit_is_control_only_and_public_prediction_is_precision_specific(
    tmp_path, monkeypatch
):
    from triton_viz.performance import GpuBackend, predict_latency
    from triton_viz.performance.calibration import stable_digest
    from triton_viz.tools import gpu_distribution_experiments as experiment
    from triton_viz.tools.gpu_cost_model_pipeline import _write, _read

    root, output = tmp_path / "run", tmp_path / "fit"
    cases, sources = [], {}
    specs = [("fp32", "ieee"), ("fp32", "tf32"), ("bf16", "ieee"), ("fp16", "ieee")]
    for width in (16, 32, 64, 128):
        for index, (dtype, precision) in enumerate(specs):
            cid = f"{width}_{index}"
            cases.append({"id": cid})
            source = dict(
                schema="triton-viz.gpu-source.v1",
                program_count=1,
                num_warps=4,
                num_stages=2,
                events=[
                    dict(
                        seq=0,
                        program=[0, 0, 0],
                        op="dot",
                        dependencies=[],
                        dtype="fp32",
                        shape=[16, 16],
                        input_shapes=[[16, width], [width, 16]],
                        elements=256,
                        dot_input_dtypes=[dtype, dtype],
                        dot_accumulator_dtype="fp32",
                        dot_input_precision=precision,
                    )
                ],
            )
            sources[cid] = source
            _write(
                root / "controls" / f"{cid}.json",
                dict(
                    role="control",
                    contaminated=False,
                    fingerprint="test",
                    cv_group=str(width),
                    source=source,
                    latency_us=2 + (index + 1) * 1e-4 * 512 * width,
                ),
            )
    _write(
        root / "manifest.json",
        dict(
            fingerprint="test",
            identity={"sm_count": 48},
            splits={"control": cases, "holdout": [{"id": "forbidden"}]},
        ),
    )
    original = experiment._read

    def guarded(path):
        assert "holdouts" not in path.parts
        return original(path)

    monkeypatch.setattr(experiment, "_read", guarded)
    experiment.fit(root, output, dot_precision=True)
    model = _read(output / "frozen_model.json")
    digest = model.pop("digest")
    assert digest == stable_digest(model)
    assert model["cv"]["passed"] and model["selection_nested_mape_pct"] <= 20
    assert "tensor_flops" not in model["feature_names"]
    assert "program_tensor_p90" not in model["feature_names"]
    backend = GpuBackend(model, "test", 48)
    for index in range(4):
        prediction = predict_latency(sources[f"32_{index}"], backend)
        assert prediction.latency_ns / 1000 == pytest.approx(
            2 + (index + 1) * 1e-4 * 512 * 32, rel=1e-5
        )
    old_source = copy.deepcopy(sources["32_0"])
    old_source["events"][0].pop("dot_input_precision")
    with pytest.raises(ValueError, match="missing_dot_precision_metadata"):
        predict_latency(old_source, backend)
