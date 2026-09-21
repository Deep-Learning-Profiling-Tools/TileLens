import hashlib
import json

import pytest

from triton_viz.tools.gpu_control_resource_audit import audit, sass_backedges


def test_resources_keep_missing_controls_and_verify_artifacts(tmp_path):
    cases = [{"id": "first"}, {"id": "second"}]
    (tmp_path / "manifest.json").write_text(
        json.dumps(dict(role="control", cases=cases))
    )
    (tmp_path / "controls").mkdir()
    ptx = "mma.sync.aligned; cp.async.ca.shared.global; ld.local.u32; fma.rn.f32;"
    row = dict(
        role="control",
        case=cases[0],
        registers_per_thread=64,
        triton_reported_spills=0,
        shared_bytes=4096,
        artifacts={"ptx": ptx, "sass": "LDL R1, [R2]; STL.64 [R3], R4;"},
        artifact_sha256={
            "ptx": hashlib.sha256(ptx.encode()).hexdigest(),
            "sass": hashlib.sha256(b"LDL R1, [R2]; STL.64 [R3], R4;").hexdigest(),
        },
    )
    path = tmp_path / "controls" / "first.json"
    path.write_text(json.dumps(row))
    report = audit(tmp_path)
    assert report["count"] == 2 and not report["complete"]
    assert report["rows"][1]["status"] == "missing"
    counts = report["rows"][0]["static_ptx_counts"]
    assert report["rows"][0]["static_sass_local_counts"] == {"loads": 1, "stores": 1}
    assert (
        counts["mma"]
        == counts["async_copy"]
        == counts["local_load"]
        == counts["fp32_fma"]
        == 1
    )
    row["artifacts"]["ptx"] += "modified"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="digest"):
        audit(tmp_path)


def test_resource_audit_rejects_non_control(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps(dict(role="holdout", cases=[])))
    with pytest.raises(ValueError, match="control"):
        audit(tmp_path)


def test_resource_audit_preserves_predicated_instruction_and_loop_counts(tmp_path):
    case = dict(id="control")
    (tmp_path / "manifest.json").write_text(
        json.dumps(dict(role="control", cases=[case]))
    )
    (tmp_path / "controls").mkdir()
    artifacts = dict(
        ptx="cp.async.ca.shared.global;",
        ttgir="scf.for %i = %lb to %ub step %s {}",
        sass="/*0000*/ @!P0 LDL.64 R1, [R2];\n/*0010*/ FFMA R1, R2, R3, R4;\n/*0020*/ @PT STL [R2], R1;",
    )
    row = dict(
        role="control",
        case=case,
        registers_per_thread=255,
        triton_reported_spills=10,
        shared_bytes=1024,
        artifacts=artifacts,
        artifact_sha256={
            k: hashlib.sha256(v.encode()).hexdigest() for k, v in artifacts.items()
        },
    )
    (tmp_path / "controls" / "control.json").write_text(json.dumps(row))
    result = audit(tmp_path)["rows"][0]
    assert result["static_ttgir_loop_count"] == 1
    assert result["static_sass_counts"] == {"LDL": 1, "STL": 1, "FFMA": 1}


def test_backedge_audit_excludes_terminal_spin_without_inventing_trip_count():
    result = sass_backedges(
        "/*0000*/ HMMA.16816.F32 R1, R2, R3, R4;\n"
        "/*0010*/ BRA.U UP0, 0x0;\n/*0020*/ BRA 0x20;"
    )
    assert result == dict(
        supported=True,
        regions=[dict(start_pc=0, branch_pc=16, static_counts={"HMMA": 1, "BRA": 1})],
    )
    assert not sass_backedges("/*0000*/ FFMA R1;\n/*0000*/ FFMA R1;")["supported"]
