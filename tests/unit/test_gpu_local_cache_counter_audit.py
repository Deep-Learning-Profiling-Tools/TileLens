import hashlib
import json

import pytest

from triton_viz.tools import gpu_local_cache_counter_audit as audit


def test_full_grid_validates_raw_geometry_and_owned_processes(tmp_path, monkeypatch):
    baseline = dict(uuid="GPU-test", driver="test", index=0)
    manifest = dict(
        role="control",
        eligible_for_fit=False,
        cases=audit.footprint_grid(),
        counter_only=True,
        iterations=65536,
        workload="local",
        metrics=list(audit.METRICS),
        baseline=baseline,
        allowed_graphics=[],
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(audit, "audit_log", lambda *a, **kw: {})
    counters = dict(
        metrics=dict(zip(audit.METRICS, [1, 1, 90, 100, 80, 20])),
        l2_hit_fraction=0.8,
        replay_count_disagreement=0,
    )
    monkeypatch.setattr(audit, "parse_counters", lambda *a: counters)
    for case in manifest["cases"]:
        p, s = case["programs"], case["local_slots"]
        stem = f"local_s{s}_p{p}_i65536"
        raw = (
            '"ID","Grid Size","Block Size","Device"\n'
            + f'"0","({p}, 1, 1)","(128, 1, 1)","0"\n'
        )
        (tmp_path / (stem + ".csv")).write_text(raw)
        (tmp_path / (stem + ".log")).write_text("raw")
        row = dict(
            **case,
            role="control",
            eligible_for_fit=False,
            status="complete",
            log_sha256=hashlib.sha256(b"raw").hexdigest(),
            counter_sha256=hashlib.sha256(raw.encode()).hexdigest(),
            native_audit={},
            counters=counters,
            monitoring=dict(
                contaminated=False,
                rejection_reasons=[],
                returncode=0,
                own_process_group=True,
                child_pid=10,
                samples=[
                    dict(
                        **baseline,
                        processes=[dict(pid=20)],
                        observed_process_groups={20: 10},
                        graphics_processes=[],
                    )
                ]
                * 2,
            ),
        )
        (tmp_path / (stem + ".json")).write_text(json.dumps(row))
    result = audit.audit_grid(tmp_path)
    assert result["count"] == 16
    assert not any(r["local_load_exact"] for r in result["rows"])
    # Mismatches are retained, not filtered to make a counter model look good.
    path = tmp_path / (stem + ".json")
    row["monitoring"]["samples"][0]["observed_process_groups"] = {20: 99}
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="Foreign process"):
        audit.audit_grid(tmp_path)
    row["monitoring"]["samples"][0]["observed_process_groups"] = {20: 10}
    path.write_text(json.dumps(row))
    (tmp_path / (stem + ".log")).write_text("changed")
    with pytest.raises(ValueError, match="Changed raw"):
        audit.audit_grid(tmp_path)
