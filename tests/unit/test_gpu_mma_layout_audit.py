import pytest

from triton_viz.tools.gpu_mma_layout_audit import (
    compiled_dot_warps,
    source_plans,
    reduction_instruction_labels,
)


def test_reduction_labels_do_not_absorb_layout_conversion_instructions():
    ptx = """
.file 1 "/control/structure.py"
.file 2 "/triton/language/standard.py"
.loc 2 191 40
shfl.sync.bfly.b32 %r0, %r1, 1, 31, -1;
bar.sync 0;
.loc 2 293 36
shfl.sync.bfly.b32 %r0, %r1, 1, 31, -1;
.loc 1 52 16
shfl.sync.bfly.b32 %r0, %r1, 1, 31, -1;
// bar.sync 0;
"""
    labels = reduction_instruction_labels(ptx)
    assert labels == dict(
        max=dict(shuffles=1, barriers=1, leading_barriers=1),
        sum=dict(shuffles=1, barriers=0, leading_barriers=0),
        other=dict(shuffles=1, barriers=0, leading_barriers=0),
    )
    with pytest.raises(ValueError, match="locations"):
        reduction_instruction_labels(ptx.replace(".loc 2 293 36", ".loc 2 999 36"))


def observation():
    return dict(
        case=dict(kind="structure_composition", variant=2, repeat=1),
        source_precision=[["fp32", "fp32", "tf32"]],
        source_features=dict(threads_per_program=128),
        ood_reasons=[],
        program_count=1,
        dot_ancestry=[
            dict(
                seq=4,
                program=[0],
                ancestor_dot_seqs=[],
                input_shapes=[[64, 32], [32, 64]],
            ),
            dict(
                seq=8,
                program=[0],
                ancestor_dot_seqs=[4],
                input_shapes=[[64, 64], [64, 64]],
            ),
        ],
    )


def test_source_chain_policy_needs_dependency_not_merely_two_dots():
    source = observation()
    assert source_plans(source, compiler_version="3.7.0") == [[4, 1], [4, 1]]
    source["dot_ancestry"][1]["ancestor_dot_seqs"] = []
    with pytest.raises(ValueError, match="producer-consumer"):
        source_plans(source, compiler_version="3.7.0")


def test_dynamic_repetition_does_not_establish_static_compiler_chain():
    source = observation()
    source["case"]["repeat"] = 5
    assert source_plans(source, compiler_version="3.7.0") is None
    source = observation()
    source["dot_ancestry"][1]["input_shapes"][1][1] = 128
    with pytest.raises(ValueError, match="Heterogeneous"):
        source_plans(source, compiler_version="3.7.0")


def test_compiler_warp_labels_do_not_enter_source_policy():
    source = observation()
    before = source_plans(source, compiler_version="3.7.0")
    for warps in ("4, 1", "1, 4"):
        text = (
            f"#mma = #ttg.nvidia_mma<{{versionMajor = 2, versionMinor = 0, warpsPerCTA = [{warps}], instrShape = [16, 8]}}>\n"
            "%x = tt.dot %a, %b, %c : ignored -> tensor<64x64xf32, #mma>\n"
        )
        assert compiled_dot_warps(text) == [[int(x) for x in warps.split(",")]]
        assert source_plans(source, compiler_version="3.7.0") == before
    with pytest.raises(ValueError, match="Missing"):
        compiled_dot_warps("")
