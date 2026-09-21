import pytest

from triton_viz.tools.gpu_cupti_perturbation_audit import audit_log


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
