import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from triton_viz.tools.gpu_packing_intervention import (
    scalarize,
    scalarize_values,
    variants,
    unroll_control_loop,
    disable_backend_unroll,
)


def test_nounroll_is_module_scoped_and_changes_only_the_directive():
    ptx = ".version 9.1\n.target sm_121a\n.address_size 64\n.visible .entry kernel() { ret; }\n"
    changed = disable_backend_unroll(ptx)
    assert changed.replace('\n.pragma "nounroll";\n', "") == ptx
    assert changed.index('.pragma "nounroll"') < changed.index(".visible .entry")
    row = dict(
        role="control",
        artifacts=dict(ptx=ptx),
        artifact_sha256=dict(ptx=hashlib.sha256(ptx.encode()).hexdigest()),
    )
    values, count = variants(row, "nounroll")
    assert values == dict(original=ptx, nounroll=changed) and count == 1
    with pytest.raises(ValueError):
        disable_backend_unroll(changed)
    with pytest.raises(ValueError):
        disable_backend_unroll(ptx.replace(".address_size 64", ".address_size 32"))
    with pytest.raises(ValueError):
        disable_backend_unroll(ptx + "\n.address_size 64\n")
    row["role"] = "holdout"
    with pytest.raises(ValueError):
        variants(row, "nounroll")


@pytest.mark.parametrize(
    "cases,mode,error",
    [
        ([], "nounroll", "nonempty unique"),
        ([dict(id="a"), dict(id="a")], "nounroll", "nonempty unique"),
        ([dict(id="a", kind="composition")], "nounroll", "pure geometry-dot"),
        ([dict(id="a", kind="geometry_dot")], "fma", "requires nounroll"),
    ],
)
def test_full_manifest_intervention_rejects_invalid_declarations(
    tmp_path, cases, mode, error
):
    from triton_viz.tools.gpu_packing_intervention import main

    (tmp_path / "manifest.json").write_text(
        json.dumps(dict(role="control", cases=cases))
    )
    output = tmp_path / "output"
    with pytest.raises(ValueError, match=error):
        main(
            [
                "--resource-root",
                str(tmp_path),
                "--output",
                str(output),
                "--ptxas",
                "must-not-execute",
                "--cuobjdump",
                "must-not-execute",
                "--mode",
                mode,
                "--all-declared-controls",
            ]
        )
    assert not output.exists()


def loop_ptx():
    return """mov.b64 %rd0, 0;
$L_loop:
fma.rn.f32x2 %rd4, %rd1, %rd2, %rd4;
add.s64 %rd0, %rd0, 4;
setp.ne.b64 %p1, %rd0, 20;
@%p1 bra $L_loop;
ret;
"""


def test_control_unrolling_checks_trip_induction_independently():
    result, trips = unroll_control_loop(loop_ptx(), 5)
    assert trips == 5
    assert result.count("fma.rn.f32x2") == 5 and "bra" not in result
    assert result.count("add.s64") == 5 and result.count("mov.b64") == 1
    assert result.endswith("ret;\n")
    with pytest.raises(ValueError, match="Declared trips"):
        unroll_control_loop(loop_ptx(), 4)


def test_unroll_rejects_extra_control_flow_or_modified_induction():
    for ptx in (
        loop_ptx().replace("add.s64", "sub.s64"),
        loop_ptx().replace("ret;", "@%p2 bra $L_loop;\nret;"),
        loop_ptx().replace("fma.rn", "$L_inner:\nfma.rn"),
        loop_ptx().replace("@%p1 bra", "mov.pred %p1, 0;\n@%p1 bra"),
    ):
        with pytest.raises(ValueError):
            unroll_control_loop(ptx, 5)


def pair_ptx():
    return """.visible .entry control() {
.reg .b64 %rd<8>;
.reg .b32 %r<12>;
mov.b64 %rd0, {%r0, %r1};
mov.b64 %rd1, %rd0;
ld.shared.v2.b64 {%rd2, %rd3}, [%r10];
fma.rn.f32x2 %rd4, %rd1, %rd2, %rd3;
mov.b64 {%r4, %r5}, %rd4;
}
"""


def test_value_pairs_split_closed_dataflow_without_narrowing_memory_access():
    result, count = scalarize_values(pair_ptx())
    assert count == 1 and "f32x2" not in result
    assert "mov.b32 %tv_pair_lo1, %tv_pair_lo0;" in result
    assert "mov.b32 %tv_pair_hi0, %r1;" in result
    assert (
        "ld.shared.v4.b32 {%tv_pair_lo2, %tv_pair_hi2, %tv_pair_lo3, %tv_pair_hi3}, [%r10];"
        in result
    )
    assert "mov.b32 %r5, %tv_pair_hi4;" in result
    assert result.count("fma.rn.f32 ") == 2


def test_value_pair_transform_refuses_pointer_arithmetic_or_predication():
    for bad in (
        pair_ptx().replace("mov.b64 %rd1, %rd0;", "add.u64 %rd1, %rd0, 4;"),
        pair_ptx().replace("fma.rn.f32x2", "@%p0 fma.rn.f32x2"),
    ):
        with pytest.raises(ValueError, match="Unsupported operation"):
            scalarize_values(bad)


def test_rewrite_preserves_lanes_and_reads_accumulator_before_alias_write():
    ptx = "fma.rn.f32x2 %rd0, %rd1, %rd2, %rd0;"
    result, count = scalarize(ptx)
    assert count == 1 and "f32x2" not in result
    assert result.count("fma.rn.f32 ") == 2
    assert "mov.b64 {%tv_scalar4, %tv_scalar5}, %rd0;" in result
    assert result.index("%tv_scalar4, %tv_scalar5") < result.index("mov.b64 %rd0,")
    assert "fma.rn.f32 %tv_scalar6, %tv_scalar0, %tv_scalar2, %tv_scalar4;" in result
    assert "fma.rn.f32 %tv_scalar7, %tv_scalar1, %tv_scalar3, %tv_scalar5;" in result


def test_unpacked_null_control_unchanged():
    ptx = "fma.rn.f32 %r0, %r1, %r2, %r0;"
    assert scalarize(ptx) == (ptx, 0)


@pytest.mark.parametrize(
    "ptx",
    [
        "@%p0 fma.rn.f32x2 %rd0, %rd1, %rd2, %rd3;",
        "fma.rn.ftz.f32x2 %rd0, %rd1, %rd2, %rd3;",
        "fma.rn.f32x2 %rd0, 0, %rd2, %rd3;",
        ".reg .b32 %tv_scalar<8>;",
    ],
)
def test_unsupported_interventions_rejected(ptx):
    with pytest.raises(ValueError):
        scalarize(ptx)


def test_target_and_modified_artifacts_rejected():
    ptx = "fma.rn.f32 %r0, %r1, %r2, %r0;"
    row = dict(
        role="holdout",
        artifacts=dict(ptx=ptx),
        artifact_sha256=dict(ptx=hashlib.sha256(ptx.encode()).hexdigest()),
    )
    with pytest.raises(ValueError, match="control"):
        variants(row)
    row["role"] = "control"
    assert variants(row)[1] == 0
    row["artifacts"]["ptx"] += "changed"
    with pytest.raises(ValueError, match="digest"):
        variants(row)


@pytest.mark.parametrize(
    "mode,all_declared", [("fma", False), ("nounroll", False), ("nounroll", True)]
)
def test_offline_runner_preserves_actual_assembler_flag_and_reproduction_checks(
    tmp_path, monkeypatch, mode, all_declared
):
    from triton_viz.tools import gpu_packing_intervention as tool

    root, output = tmp_path / "controls", tmp_path / "result"
    (root / "controls").mkdir(parents=True)
    cases = [
        dict(id=f"resource_dot_float32_ieee_128x256x32_w{w}_s{s}_{r}")
        for w, s, r in ((4, 1, 1), (4, 1, 5), (4, 2, 5), (8, 1, 5))
    ]
    if mode == "nounroll":
        cases = [
            dict(id=f"pressure_p48_{dtype}_{precision}_{m}x128x32_w{warps}_s2")
            for dtype, precision, m, warps in (
                ("bfloat16", "ieee", 256, 4),
                ("float32", "tf32", 128, 4),
                ("float32", "tf32", 128, 8),
                ("float32", "ieee", 256, 4),
            )
        ]
    if all_declared:
        cases = [dict(id="independent_control", kind="geometry_dot")]
    (root / "manifest.json").write_text(json.dumps(dict(role="control", cases=cases)))
    ptx = ".target sm_121a\n.address_size 64\nfma.rn.f32x2 %rd0, %rd1, %rd2, %rd0;\n"
    for case in cases:
        row = dict(
            role="control",
            case=case,
            artifacts=dict(ptx=ptx, sass="SASS"),
            artifact_sha256=dict(ptx=hashlib.sha256(ptx.encode()).hexdigest()),
            cubin_sha256=hashlib.sha256(b"binary").hexdigest(),
        )
        (root / "controls" / (case["id"] + ".json")).write_text(json.dumps(row))
    assembler = tmp_path / "ptxas"
    assembler.write_bytes(b"assembler identity")
    invocations = []

    def run(command, **kwargs):
        invocations.append(command)
        if "-o" in command:
            assert "--regAllocOptLevel=2" in command
            Path(command[-1]).write_bytes(b"binary")
        return SimpleNamespace(stdout="SASS" if "--dump-sass" in command else "13.1")

    monkeypatch.setattr(tool.subprocess, "run", run)
    tool.main(
        [
            "--resource-root",
            str(root),
            "--output",
            str(output),
            "--ptxas",
            str(assembler),
            "--cuobjdump",
            "cuobjdump",
            "--regalloc-opt-level",
            "2",
            "--mode",
            mode,
            *(["--all-declared-controls"] if all_declared else []),
        ]
    )
    assert sum("-o" in c for c in invocations) == 2 * len(cases)
    report = json.loads((output / cases[0]["id"] / "original.json").read_text())
    assert report["matches_archived_cubin"] and report["matches_archived_sass"]
    assert report["numerical_validation"] == "not executed"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["regalloc_opt_level"] == 2 and not manifest["gpu_execution"]
    assert not manifest["eligible_for_fit"]
    assert manifest["intervention"] == mode
    assert manifest["all_declared_controls"] == all_declared
    assert manifest["cases"] == cases
    variant = "nounroll" if mode == "nounroll" else "scalar_fma"
    changed_report = json.loads(
        (output / cases[0]["id"] / (variant + ".json")).read_text()
    )
    assert changed_report["nounroll_directives"] == int(mode == "nounroll")
    assert changed_report["rewritten_fmas"] == int(mode == "fma")
    original_path = root / "controls" / (cases[0]["id"] + ".json")
    changed = json.loads(original_path.read_text())
    changed["cubin_sha256"] = "different baseline"
    original_path.write_text(json.dumps(changed))
    failed = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="does not reproduce"):
        tool.main(
            [
                "--resource-root",
                str(root),
                "--output",
                str(failed),
                "--ptxas",
                str(assembler),
                "--cuobjdump",
                "cuobjdump",
                "--regalloc-opt-level",
                "2",
                "--mode",
                mode,
                *(["--all-declared-controls"] if all_declared else []),
            ]
        )
    assert (failed / cases[0]["id"] / "original.json").exists()
    assert not (failed / cases[0]["id"] / (variant + ".ptx")).exists()
