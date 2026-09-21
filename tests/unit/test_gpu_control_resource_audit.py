import hashlib
import json

import pytest

from triton_viz.tools.gpu_control_resource_audit import audit


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
