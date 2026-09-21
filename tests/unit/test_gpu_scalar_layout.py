import copy

import pytest

from triton_viz.performance.gpu_scalar_layout import scalar_layout_work
from triton_viz.performance.triton_observe import observe


@pytest.mark.parametrize("m,sfu,shuffles", [(32, 12, 22), (64, 34, 8)])
def test_chained_normalization_uses_layout_and_row_broadcast(
    m, sfu, shuffles, monkeypatch
):
    import triton
    import triton_viz
    from microbench.gpu.tests.coverage.kernels import prepare, check_output

    def forbidden(*args, **kwargs):
        raise AssertionError("Source mapping must not compile")

    monkeypatch.setattr(triton, "compile", forbidden)
    case = dict(
        kind="structure_composition",
        programs=1,
        bm=m,
        bn=64,
        bk=32,
        dtype="bfloat16",
        precision="ieee",
        num_warps=8,
        num_stages=1,
        repeat=1,
        variant=2,
    )
    kernel, grid, args, output = prepare(case, "cpu")
    try:
        source = observe(
            kernel, grid, *args, num_warps=8, num_stages=1, capture_loops=True
        )
        check_output(case, output)
        options = dict(precision=[["bf16", "bf16", "ieee"]], compiler_version="3.7.0")
        work = scalar_layout_work(source, **options)
        assert work["sfu_warp_instructions"] == sfu * 8
        assert work["shuffle_warp_instructions"] == shuffles * 8
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
