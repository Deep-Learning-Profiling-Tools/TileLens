import json
import hashlib

import pytest

from triton_viz.tools.gpu_control_resources import selected_controls
from triton_viz.tools.gpu_local_counter_collect import (
    METRICS,
    commands,
    reusable_controls,
)


def test_collects_all_declared_controls_without_timing_or_clock_changes(tmp_path):
    result = commands("ncu", tmp_path)
    assert len(result) == 32
    assert {c[c.index("--case-id") + 1] for c in result} == {
        c["id"] for c in selected_controls("pressure")
    }
    for command in result:
        for flag, value in (
            ("--clock-control", "none"),
            ("--cache-control", "all"),
            ("--replay-mode", "kernel"),
            ("--profile-from-start", "off"),
            ("--metrics", ",".join(METRICS)),
            ("--suite", "pressure"),
        ):
            assert command[command.index(flag) + 1] == value
        assert "--allow-idle-graphics" not in command
    assert all(
        "--allow-idle-graphics" in c
        for c in commands("ncu", tmp_path, allow_idle_graphics=True)
    )


def test_pipeline_counter_grid_keeps_every_new_control(tmp_path):
    declared = selected_controls("pressure_pipeline")
    result = commands("ncu", tmp_path, suite="pressure_pipeline")
    assert len(result) == 32
    assert {c[c.index("--case-id") + 1] for c in result} == {c["id"] for c in declared}
    assert all(c[c.index("--suite") + 1] == "pressure_pipeline" for c in result)
    with pytest.raises(ValueError):
        commands("ncu", tmp_path, suite="holdout")


def test_monitored_collection_does_not_relabel_inherited_unmonitored_data(tmp_path):
    from triton_viz.tools.gpu_local_counter_collect import main

    (tmp_path / "parent").mkdir()
    (tmp_path / "parent" / "manifest.json").write_text(
        json.dumps(dict(monitored=False))
    )
    with pytest.raises(ValueError, match="cannot inherit"):
        main(
            [
                "--ncu",
                "ncu",
                "--output",
                str(tmp_path / "new"),
                "--monitor",
                "--resume-from",
                str(tmp_path / "parent"),
            ]
        )


def test_resume_reuses_only_complete_matching_control_rows(tmp_path):
    cases = selected_controls("pressure")
    hardware = dict(uuid="gpu", driver="driver", index=0)
    manifest = dict(
        role="control",
        cases=cases,
        metrics=list(METRICS),
        hardware=hardware,
        probe_sha256="probe",
        ncu_version="ncu",
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    case = cases[0]
    path = tmp_path / (case["id"] + ".csv")
    path.write_text(
        "\n".join(
            ['"ID","Kernel Name","Metric Name","Metric Value"']
            + [f'"0","geometry_dot","{m}","0"' for m in METRICS]
        )
    )
    path.with_suffix(".log").write_text(
        f"control={case['id']} numerical=passed profiler_timing_not_for_fit"
    )
    kwargs = dict(hardware=hardware, probe_sha256="probe", ncu_version="ncu")
    assert set(reusable_controls(tmp_path, **kwargs)) == {case["id"]}
    with pytest.raises(ValueError, match="unmonitored"):
        reusable_controls(tmp_path, **kwargs, require_monitored=True)
    manifest.update(monitored=True, allowed_graphics=[])
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    record = dict(
        case_id=case["id"],
        csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        log_sha256=hashlib.sha256(path.with_suffix(".log").read_bytes()).hexdigest(),
        monitoring=dict(
            returncode=0,
            contaminated=False,
            rejection_reasons=[],
            child_pid=10,
            own_process_group=True,
            samples=[dict(**hardware, processes=[], graphics_processes=[])] * 2,
        ),
    )
    path.with_suffix(".monitor.json").write_text(json.dumps(record))
    reused = reusable_controls(
        tmp_path, **kwargs, require_monitored=True, allowed_graphics=[]
    )
    assert set(reused) == {case["id"]}
    assert "monitor_sha256" in reused[case["id"]]
    record["monitoring"]["contaminated"] = True
    path.with_suffix(".monitor.json").write_text(json.dumps(record))
    assert not reusable_controls(tmp_path, **kwargs, require_monitored=True)
    assert not reusable_controls(tmp_path, **kwargs)  # Cannot downgrade known failures.
    path.with_suffix(".log").write_text("BusyGPU: foreign process")
    assert not reusable_controls(tmp_path, **kwargs)
    with pytest.raises(ValueError, match="different control"):
        reusable_controls(tmp_path, **{**kwargs, "probe_sha256": "changed"})
