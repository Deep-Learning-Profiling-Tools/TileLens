import copy

import pytest
import triton
import triton.language as tl

from microbench.gpu.tests.coverage.geometry import prepare, check_output
from triton_viz.performance.gpu_memory import memory_features
from triton_viz.performance.triton_observe import observe


@triton.jit
def _masked_repeat(X, Y, N: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, 64)
    a = tl.load(X + offsets, offsets < N, other=0)
    b = tl.load(X + offsets, offsets < N, other=0)
    tl.store(Y + pid * 64 + offsets, a + b, offsets < N)


def test_working_set_masks_and_within_program_repeated_loads():
    import torch
    import triton_viz

    x = torch.arange(33, dtype=torch.float32)
    y = torch.full((2, 64), -9.0)
    try:
        source = observe(_masked_repeat, (2,), x, y, 33)
        summary = source["memory_working_set"]
        assert summary["load_unique_sectors"] == 5
        assert summary["load_sector_requests"] == 20
        assert summary["store_sector_requests"] == 10
        assert summary["program_load_sectors_p90"] == 10
        features, reasons = memory_features(source)
        assert not reasons and features["load_repeat_sectors"] == 15
        torch.testing.assert_close(y[:, :33], (2 * x).expand(2, -1))
        assert torch.all(y[:, 33:] == -9)
    finally:
        triton_viz.clear()


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
def test_unique_working_set_counts_reuse_without_exposing_addresses(dtype, monkeypatch):
    import torch
    import triton
    import triton_viz

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU observation must not compile or initialize CUDA")

    monkeypatch.setattr(triton.compiler, "compile", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    size = 4 if dtype == "float32" else 2
    results = []
    try:
        for reuse in ("none", "a", "ab"):
            case = dict(
                programs=2,
                repeat=2,
                bm=32,
                bn=32,
                bk=32,
                dtype=dtype,
                precision="ieee",
                reuse=reuse,
            )
            kernel, grid, args, out = prepare(case, "cpu")
            source = observe(kernel, grid, *args)
            check_output(case, out)
            features, reasons = memory_features(source)
            assert not reasons
            one_operand = 2 * 32 * 32 * size // 32
            factor = {"none": 4, "a": 3, "ab": 2}[reuse]
            assert features["load_unique_sectors"] == factor * one_operand
            assert features["load_repeat_sectors"] == (4 - factor) * one_operand
            assert features["store_sector_requests"] == 2 * 32 * 32 * 4 // 32
            assert set(source["memory_working_set"]) == {
                "schema",
                "load_unique_sectors",
                "load_sector_requests",
                "store_sector_requests",
                "program_load_sectors_p90",
            }
            assert (
                source["memory_working_set"]["program_load_sectors_p90"]
                == 2 * one_operand
            )
            results.append(features)
            triton_viz.clear()
        assert (
            results[0]["load_footprint_program_pressure"]
            == 2 * results[2]["load_footprint_program_pressure"]
        )
        legacy = copy.deepcopy(source)
        legacy.pop("memory_working_set")
        assert memory_features(legacy)[1] == ["missing_memory_working_set_metadata"]
        source["memory_working_set"]["load_unique_sectors"] = float("nan")
        assert memory_features(source)[1] == ["invalid_memory_working_set_metadata"]
    finally:
        triton_viz.clear()


@pytest.mark.parametrize("wave_dot", [False, True])
def test_memory_fit_is_control_only_and_public_prediction_uses_reuse(
    tmp_path, wave_dot
):
    from triton_viz.tools.gpu_fit_guarded import guarded_fit
    from triton_viz.tools.gpu_cost_model_pipeline import _write, _read
    from triton_viz.performance import GpuBackend, predict_latency
    from triton_viz.performance.calibration import stable_digest

    root, output = tmp_path / "run", tmp_path / "fit"
    cases, sources = [], []
    for width in (32, 64, 128, 256):
        for sharing in (1, 2, 4):
            cid = f"w{width}_s{sharing}"
            requests, unique, stores = width * 8, width * 8 // sharing, width
            source = dict(
                schema="triton-viz.gpu-source.v1",
                program_count=1,
                num_warps=4,
                num_stages=2,
                memory_working_set=dict(
                    schema="triton-viz.gpu-memory-working-set.v1",
                    load_unique_sectors=unique,
                    load_sector_requests=requests,
                    store_sector_requests=stores,
                    program_load_sectors_p90=requests,
                ),
                events=[
                    dict(
                        seq=i,
                        op=op,
                        program=[0],
                        dependencies=[],
                        dtype="fp32",
                        elements=width,
                        input_shapes=[],
                        shape=[width],
                        sectors=sectors,
                    )
                    for i, (op, sectors) in enumerate(
                        (("load", requests), ("store", stores))
                    )
                ],
            )
            latency = 2 + unique * 0.01 + (requests - unique) * 0.001 + stores * 0.005
            cases.append({"id": cid})
            sources.append((source, latency))
            _write(
                root / "controls" / (cid + ".json"),
                dict(
                    role="control",
                    contaminated=False,
                    fingerprint="test",
                    cv_group=str(width),
                    source=source,
                    latency_us=latency,
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
    guarded_fit(root, output, wave_dot=wave_dot)
    model = _read(output / "frozen_model.json")
    digest = model.pop("digest")
    assert digest == stable_digest(model)
    assert model["feature_set"].startswith("dot_precision_memory")
    assert model["cv"]["passed"] and model["selection_nested_mape_pct"] <= 20
    assert len(_read(output / "fit_reads.json")) == 13
    backend = GpuBackend(model, "test", 48)
    for source, expected in sources:
        prediction = predict_latency(source, backend)
        assert prediction.latency_ns / 1000 == pytest.approx(expected, rel=1e-5)
    source = copy.deepcopy(sources[0][0])
    source.pop("memory_working_set")
    with pytest.raises(ValueError, match="missing_memory_working_set_metadata"):
        predict_latency(source, backend)
