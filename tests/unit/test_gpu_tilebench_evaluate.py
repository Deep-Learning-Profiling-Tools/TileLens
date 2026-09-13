from triton_viz.tools.gpu_tilebench_evaluate import Capture, cases, summarize
from triton_viz.tools.gpu_cost_model_pipeline import _write


def test_formal_suite_deduplicates_to_254():
    from pathlib import Path

    rows = cases(Path("microbench/inf2_nki/configs/formal_holdouts.json"))
    assert len(rows) == len({r["id"] for r in rows}) == 254
    assert sum(r["op"] == "tiled_attention" for r in rows) == 4
    assert sum(r["op"] == "matmul_fp32_fp16_fp8" for r in rows) == 10


def test_capture_does_not_compile_and_records_defaults():
    launches = []
    kernel = object()
    Capture(kernel, launches)[(3,)]("tensor", BLOCK_SIZE=1024)
    assert launches == [
        (kernel, (3,), ("tensor",), dict(BLOCK_SIZE=1024, num_warps=4, num_stages=3))
    ]


def test_partial_report_cannot_claim_full_mape(tmp_path):
    declared = [dict(id="a", op="relu"), dict(id="b", op="relu")]
    _write(
        tmp_path / "cases/a.json",
        dict(case=declared[0], error_pct=10, ood_reasons=["tile"]),
    )
    _write(tmp_path / "cases/b.json", dict(case=declared[1], error="unsupported"))
    report = summarize(tmp_path, declared)
    assert report["full_254_mape_pct"] is None
    assert report["in_domain_mape_pct"] is None
    assert report["diagnostic_mape_pct"] == 10
    assert report["scored"] == 1
    assert len(report["failures"]) == 1


def test_bf16_scalar_compat_is_scoped_and_rounds():
    import numpy as np
    import torch
    from triton.runtime.interpreter import InterpreterBuilder
    from triton_viz.performance.triton_observe import _bf16_constant_compat

    original = getattr(InterpreterBuilder, "get_bf16", None)
    with _bf16_constant_compat():
        builder = InterpreterBuilder()
        for value in (2.0, -1.25, 1.00390625, 1.01171875):
            result = builder.get_bf16(value)
            expected = torch.tensor([value]).bfloat16().view(torch.uint16).numpy()
            np.testing.assert_array_equal(result.data, expected)
    assert getattr(InterpreterBuilder, "get_bf16", None) is original


def test_bf16_interpreter_arithmetic_uses_values_not_storage_bits():
    import numpy as np
    import torch
    import triton.language as tl
    from triton.runtime.interpreter import TensorHandle
    from triton_viz.core.frontend.base import get_frontend
    from triton_viz.performance.triton_observe import _bf16_constant_compat

    x = torch.tensor([[1.5, -2.0], [0.25, 3.0]]).bfloat16()
    handle = TensorHandle(x.view(torch.uint16).numpy(), tl.bfloat16)
    frontend = get_frontend("triton")
    operations = frontend.original_ops[frontend.builder]
    original = dict(operations)
    with _bf16_constant_compat():
        product = operations["binary_op"](handle, handle, np.multiply)
        np.testing.assert_array_equal(product.data, (x * x).view(torch.uint16).numpy())
        accumulator = TensorHandle(np.zeros((2, 2), np.float32), tl.float32)
        dot = operations["create_dot"](handle, handle, accumulator, "ieee", 0)
        np.testing.assert_allclose(dot.data, (x.float() @ x.float()).numpy())
    assert operations == original


def test_attention_adapter_cpu_reference():
    from pathlib import Path
    import torch
    import triton
    import pytest
    from triton_viz.tools.gpu_tilebench_evaluate import prepare
    from triton_viz.performance.triton_observe import observe

    if tuple(int(v) for v in triton.__version__.split(".")[:2]) < (3, 7):
        pytest.skip("Repository interpreter patching requires Triton 3.7+")

    for width in (64, 512):
        case = dict(op="tiled_attention", rows=128, cols=width, dtype="float32")
        (kernel, grid, args, kwargs), output, reference = prepare(
            case, Path("."), adapters=True
        )
        source = observe(kernel, grid, *args, **kwargs)
        assert source["program_count"] == 4 * width // 64
        torch.testing.assert_close(output, reference, rtol=0.01, atol=0.002)


def test_merge_keeps_first_accepted_measurement(tmp_path):
    from triton_viz.tools.gpu_tilebench_report import merge

    case = dict(id="mul2", op="mul2", rows=1, cols=128, dtype="bfloat16")
    primary, supplement = tmp_path / "primary", tmp_path / "supplement"
    manifest = dict(
        cases=[case],
        calibration_digest="frozen",
        calibration_fingerprint="same",
        tilebench_sources={},
    )
    for root in (primary, supplement):
        _write(root / "manifest.json", manifest)
    _write(
        primary / "cases/mul2.json",
        dict(case=case, measured_us=2.0, error="observation failed"),
    )
    _write(
        supplement / "cases/mul2.json",
        dict(
            case=case,
            measured_us=1.0,
            predicted_us=1.0,
            error_pct=0.0,
            ood_reasons=["dtype"],
            implementation="unmodified_tilebench",
        ),
    )
    report = merge(primary, supplement, tmp_path / "merged")
    assert report["diagnostic_mape_pct"] == 50.0
