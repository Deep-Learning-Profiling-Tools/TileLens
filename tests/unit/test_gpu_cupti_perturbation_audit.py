import pytest
import json
import hashlib

from triton_viz.tools.gpu_cupti_perturbation_audit import audit_log


def test_matrix_audit_requires_all_trials_and_checks_raw_hashes(tmp_path, monkeypatch):
    from triton_viz.tools import gpu_cupti_perturbation_audit as module
    from triton_viz.tools.gpu_cupti_perturbation_collect import declared_trials

    cases = declared_trials()
    baseline = dict(uuid="GPU-test", driver="test", index=0)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            dict(
                role="control",
                eligible_for_fit=False,
                trials=cases,
                baseline=baseline,
                allowed_graphics=[],
            )
        )
    )
    result = dict(body_envelope=dict(median_us=1.0), cupti_kernel=dict(median_us=2.0))
    monkeypatch.setattr(module, "audit_log", lambda *a, **k: result)
    for case in cases:
        (tmp_path / (case["id"] + ".log")).write_text("raw")
        (tmp_path / (case["id"] + ".json")).write_text(
            json.dumps(
                dict(
                    role="control",
                    eligible_for_fit=False,
                    case=case,
                    status="complete",
                    log_sha256=hashlib.sha256(b"raw").hexdigest(),
                    audit=result,
                    monitoring=dict(
                        contaminated=False,
                        rejection_reasons=[],
                        returncode=0,
                        child_pid=123,
                        samples=[dict(**baseline, processes=[], graphics_processes=[])]
                        * 2,
                    ),
                )
            )
        )
    audited = module.audit_root(tmp_path)
    assert audited["count"] == 48 and len(audited["paired"]) == 6
    assert all(len(g["pairs"]) == 4 for g in audited["paired"])
    (tmp_path / (cases[-1]["id"] + ".log")).write_text("changed")
    with pytest.raises(ValueError, match="changed trial"):
        module.audit_root(tmp_path)


def log(mode="software_serial"):
    lines = [
        f"role=control,eligible_for_fit=false,mode={mode},programs=48,iterations=16",
        "l2_bytes=1024,sm_count=48,eviction_bytes=2048",
        "warmup_launches=1",
    ]
    if mode == "software_serial":
        for ordinal, sample in enumerate([-1] + list(range(352))):
            for index, name in enumerate(
                ("perturbation_eviction", "perturbation_body")
            ):
                start = 1 + ordinal * 1000 + index * 100
                lines.append(f"cupti,{sample},{index},{start},{start+90},0,1,1,{name}")
            lines.append(f"delivery,{sample},2,0,1")
    for sample in range(352):
        for block in range(48):
            start = 10000 + sample * 1000 + block
            lines.append(f"body,{sample},{block},{start},{start+10}")
    lines.append("completed=352,valid=1,close_status=0,eligible_for_fit=false")
    return "\n".join(lines)


@pytest.mark.parametrize("mode", ["none", "software_serial"])
def test_retains_all_groups_and_separates_clock_domains(mode):
    result = audit_log(log(mode), mode=mode, programs=48, iterations=16)
    assert result["eligible_for_fit"] is False
    assert len(result["body_envelope"]["samples_us"]) == 352
    assert len(result["body_envelope"]["group_means_us"]) == 11
    assert result["body_envelope"]["median_us"] == pytest.approx(0.057)
    if mode == "software_serial":
        assert result["cupti_minus_body_us"] == pytest.approx([0.033] * 352)


@pytest.mark.parametrize(
    "before,after",
    [
        ("delivery,0,2,0,1", "delivery,0,2,1,1"),
        ("completed=352", "completed=351"),
        ("valid=1,close_status", "valid=0,close_status"),
        ("body,0,0,10000,10010\n", ""),
        ("warmup_launches=1", "warmup_launches=2"),
        ("eviction_bytes=2048", "eviction_bytes=1024"),
        ("role=control", "role=holdout"),
        (",0,1,1,perturbation_body", ",0,2,1,perturbation_body"),
    ],
)
def test_incomplete_contaminated_or_inconsistent_logs_are_not_accepted(before, after):
    with pytest.raises(ValueError):
        audit_log(
            log().replace(before, after),
            mode="software_serial",
            programs=48,
            iterations=16,
        )
