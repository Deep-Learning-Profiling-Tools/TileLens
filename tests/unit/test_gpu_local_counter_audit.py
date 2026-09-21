import json
import hashlib

import pytest

from triton_viz.tools.gpu_local_counter_audit import audit, parse
from triton_viz.tools.gpu_local_counter_collect import METRICS
from triton_viz.tools.gpu_control_resources import selected_controls


def test_process_group_monitor_audit_checks_ownership_and_raw_hashes(tmp_path):
    from triton_viz.tools.gpu_local_counter_audit import validate_monitor

    path = tmp_path / "control.csv"
    path.write_text("csv")
    path.with_suffix(".log").write_text("log")
    hardware = dict(uuid="GPU-test", driver="test", index=0)
    manifest = dict(hardware=hardware, allowed_graphics=[])
    record = dict(
        case_id="control",
        csv_sha256=hashlib.sha256(b"csv").hexdigest(),
        log_sha256=hashlib.sha256(b"log").hexdigest(),
        monitoring=dict(
            returncode=0,
            contaminated=False,
            rejection_reasons=[],
            own_process_group=True,
            child_pid=10,
            samples=[
                dict(
                    **hardware,
                    processes=[dict(pid=20)],
                    observed_process_groups={20: 10},
                    graphics_processes=[],
                )
            ]
            * 2,
        ),
    )
    path.with_suffix(".monitor.json").write_text(json.dumps(record))
    validate_monitor(path, manifest)
    record["monitoring"]["samples"][0]["observed_process_groups"] = {20: 30}
    path.with_suffix(".monitor.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Foreign process"):
        validate_monitor(path, manifest)
    path.write_text("changed")
    with pytest.raises(ValueError, match="hash or identity"):
        validate_monitor(path, manifest)


def table(values=(0, 32)):
    return "\n".join(
        ['"ID","Kernel Name","Metric Name","Metric Value"']
        + [
            f'"0","geometry_dot","{metric}","{value}"'
            for metric, value in zip(METRICS, values)
        ]
    )


def test_zero_local_traffic_is_valid_and_both_counters_required():
    assert parse(table()) == {"LDL": 0, "STL": 32}
    with pytest.raises(ValueError):
        parse(table((0,)))


@pytest.mark.parametrize(
    "text",
    [
        table((float("nan"), 32)),
        table((-1, 32)),
        table((1.5, 32)),
        table().replace('"geometry_dot"', '"target"'),
        table().replace('"0","geometry_dot"', '"1","geometry_dot"'),
        table() + "\n" + table().splitlines()[-1],
    ],
)
def test_reject_invalid_or_unexpected_counters(text):
    with pytest.raises(ValueError):
        parse(text)


def test_missing_controls_retained_and_holdout_manifest_rejected(tmp_path):
    counters, resources = tmp_path / "counters", tmp_path / "resources"
    counters.mkdir()
    resources.mkdir()
    manifest = dict(
        role="control",
        cases=selected_controls("pressure"),
        hardware=dict(uuid="test-gpu"),
        metrics=list(METRICS),
    )
    for path in (counters, resources):
        (path / "manifest.json").write_text(json.dumps(manifest))
    result = audit(counters, resources)
    assert result["count"] == 32 and not result["complete"]
    assert result["compared_count"] == 0 and not result["eligible_for_fit"]
    assert all(r["counter_status"] == "incomplete" for r in result["rows"])
    manifest["role"] = "holdout"
    (resources / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="control manifests"):
        audit(counters, resources)


def test_counter_labels_cannot_change_dot_work_trip_hypothesis(tmp_path, monkeypatch):
    from triton_viz.tools import gpu_dot_lowering_audit, gpu_local_counter_audit

    counters, resources = tmp_path / "counters", tmp_path / "resources"
    counters.mkdir()
    (resources / "controls").mkdir(parents=True)
    cases = selected_controls("pressure")
    manifest = dict(
        role="control", cases=cases, hardware=dict(uuid="gpu"), metrics=list(METRICS)
    )
    for root in (counters, resources):
        (root / "manifest.json").write_text(json.dumps(manifest))
    case = cases[0]
    (resources / "controls" / (case["id"] + ".json")).write_text(
        json.dumps(dict(case=case))
    )
    path = counters / (case["id"] + ".csv")
    path.with_suffix(".log").write_text(
        f"control={case['id']} numerical=passed profiler_timing_not_for_fit"
    )
    monkeypatch.setattr(
        gpu_dot_lowering_audit,
        "audit",
        lambda *_: dict(
            rows=[
                dict(
                    case=c,
                    applicable=True,
                    dot_work_loop_hypothesis=dict(supported=True, loop_trips=3),
                )
                for c in cases
            ]
        ),
    )
    monkeypatch.setattr(
        gpu_local_counter_audit,
        "account",
        lambda _, loop_trips: dict(
            assumed_loop_trips=loop_trips,
            conditional_payload_sector_equivalents=dict(LDL=0, STL=0),
        ),
    )
    predictions = []
    for values in ((0, 0), (999, 999)):
        path.write_text(table(values))
        result = audit(counters, resources, tmp_path / "source")
        predictions.append(result["rows"][0]["conditional_accounting"])
    assert predictions[0] == predictions[1]
    assert predictions[0]["assumed_loop_trips"] == 3
