import copy

import pytest

from triton_viz.performance.gpu_scalar_layout import scalar_layout_work
from triton_viz.performance.triton_observe import observe


@pytest.mark.parametrize(
    "m,warps,mode,sfu,shuffles",
    [
        (32, 4, 1, 20, 20),
        (32, 4, 2, 20, 20),
        (64, 4, 1, 36, 18),
        (64, 4, 2, 34, 8),
        (32, 8, 1, 10, 12),
        (32, 8, 2, 12, 22),
        (64, 8, 1, 20, 20),
        (64, 8, 2, 34, 8),
    ],
)
@pytest.mark.parametrize(
    "dtype,precision,semantic",
    [
        ("bfloat16", "ieee", "bf16"),
        ("float16", "ieee", "fp16"),
        ("float32", "tf32", "fp32"),
    ],
)
def test_chained_normalization_uses_layout_and_row_broadcast(
    m, warps, mode, sfu, shuffles, dtype, precision, semantic, monkeypatch
):
    import torch
    import triton
    import triton_viz
    from microbench.gpu.tests.coverage.kernels import prepare, check_output

    def forbidden(*args, **kwargs):
        raise AssertionError("Source mapping must not compile")

    monkeypatch.setattr(triton, "compile", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    case = dict(
        kind="structure_composition",
        programs=1,
        bm=m,
        bn=64,
        bk=32,
        dtype=dtype,
        precision=precision,
        num_warps=warps,
        num_stages=1,
        repeat=1,
        variant=mode,
    )
    kernel, grid, args, output = prepare(case, "cpu")
    try:
        source = observe(
            kernel, grid, *args, num_warps=warps, num_stages=1, capture_loops=True
        )
        check_output(case, output)
        options = dict(
            precision=[[semantic, semantic, precision]], compiler_version="3.7.0"
        )
        work = scalar_layout_work(source, **options)
        assert work["sfu_warp_instructions"] == sfu * warps
        assert work["shuffle_warp_instructions"] == shuffles * warps
        assert not work["compiler_regions_verified"]
        missing = copy.deepcopy(source)
        for event in missing["events"]:
            event.pop("reduction_axis", None)
        with pytest.raises(ValueError, match="axis-1"):
            scalar_layout_work(missing, **options)
        reversed_division = copy.deepcopy(source)
        for event in reversed_division["events"]:
            if event.get("primitive") == "divide":
                event["operand_dependencies"].reverse()
        with pytest.raises(ValueError, match="divisor"):
            scalar_layout_work(reversed_division, **options)
    finally:
        triton_viz.clear()
