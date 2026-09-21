import hashlib
import json

import pytest

from triton_viz.tools.gpu_dot_lowering_audit import audit, dot_work_loop_hypothesis


def test_late_unroll_trip_hypothesis_uses_dot_work_not_traffic():
    cfg = dict(supported=True, regions=[dict(static_counts={"HMMA": 80})])
    result = dot_work_loop_hypothesis(cfg, "HMMA", 112, 272)
    assert result["supported"] and result["loop_trips"] == 3
    assert result["outside_instructions"] == 32
    assert not dot_work_loop_hypothesis(cfg, "HMMA", 112, 273)["supported"]
    assert not dot_work_loop_hypothesis(cfg, "HMMA", 112, 31)["supported"]


def test_fully_unrolled_and_multiple_loop_boundaries():
    cfg = dict(supported=True, regions=[])
    assert dot_work_loop_hypothesis(cfg, "HMMA", 80, 80)["supported"]
    assert not dot_work_loop_hypothesis(cfg, "HMMA", 16, 80)["supported"]
    cfg["regions"] = [dict(), dict()]
    assert not dot_work_loop_hypothesis(cfg, "HMMA", 80, 80)["supported"]


@pytest.mark.parametrize(
    "precision,opcode,count", [("fp16", "HMMA", 16), ("fp32", "FFMA", 1024)]
)
def test_instruction_expansion_is_checked_not_assumed(
    tmp_path, precision, opcode, count
):
    resources, sources = tmp_path / "resources", tmp_path / "sources"
    case = dict(id="control", kind="geometry_dot")
    for root in (resources, sources):
        (root / "controls").mkdir(parents=True)
        (root / "manifest.json").write_text(
            json.dumps(dict(role="control", cases=[case]))
        )
    ptx = (
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32;"
        if precision == "fp16"
        else "fma.rn.f32;"
    )
    artifacts = dict(
        ptx="\n".join([ptx] * count),
        ttgir="scf.for %i = %lo to %hi step %s {}",
        sass="\n".join(
            f"/*{i * 16:04x}*/ {opcode} R1, R2, R3, R4;" for i in range(count)
        ),
    )
    compiled = dict(
        role="control",
        case=case,
        artifacts=artifacts,
        artifact_sha256={
            k: hashlib.sha256(v.encode()).hexdigest() for k, v in artifacts.items()
        },
        registers_per_thread=128,
        triton_reported_spills=0,
        shared_bytes=0,
    )
    (resources / "controls" / "control.json").write_text(json.dumps(compiled))
    source = dict(
        role="control",
        case=case,
        numerical_validation="passed",
        compile_and_cuda_forbidden=True,
        dot_shapes=[[[64, 32], [32, 64]]],
        ood_reasons=[],
        operation_counts=dict(dot=240),
        program_count=48,
        source_features=dict(threads_per_program=128),
        source_precision=[[precision, precision, "ieee"]],
    )
    (sources / "controls" / "control.json").write_text(json.dumps(source))
    result = audit(resources, sources)
    assert result["checked_count"] == result["exact_count"] == 1
    assert result["ptx_exact_count"] == 1
    assert result["rows"][0]["predicted_static_instructions"] == count
    assert result["eligible_for_fit"] is False
    # LLVM may further unroll after TTGIR; retain this mismatch, do not correct
    # the prediction by reading emitted SASS counts into its formula.
    artifacts["sass"] += f"\n/*ffff*/ {opcode} R1, R2, R3, R4;"
    compiled["artifact_sha256"]["sass"] = hashlib.sha256(
        artifacts["sass"].encode()
    ).hexdigest()
    (resources / "controls" / "control.json").write_text(json.dumps(compiled))
    result = audit(resources, sources)
    assert result["checked_count"] == 1 and result["exact_count"] == 0
