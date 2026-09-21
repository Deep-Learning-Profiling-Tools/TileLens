import json

import pytest

from triton_viz.tools.gpu_pressure_mechanism_audit import audit


@pytest.fixture
def diagnostics(tmp_path):
    cases = [dict(id=f"control_w{w}", num_warps=w, bm=256) for w in (4, 8)]
    (tmp_path / "manifest.json").write_text(
        json.dumps(dict(role="control", cases=cases))
    )
    (tmp_path / "controls").mkdir()
    resources = []
    for case in cases:
        warps = case["num_warps"]
        row = dict(
            role="control",
            case=case,
            eligible_for_fit=False,
            samples=[dict(latency_us=100 / warps)] * 11,
            median_us=100 / warps,
            relative_span=0,
            unstable=False,
            graph_kernel_nodes=22,
            dropped_records=0,
            numerical_validation="passed",
            source=dict(num_warps=warps, num_stages=1, events=[]),
        )
        directory = tmp_path / "attempts" / case["id"]
        directory.mkdir(parents=True)
        (directory / "1.json").write_text(json.dumps(row))
        (tmp_path / "controls" / f"{case['id']}.json").write_text(
            json.dumps({**row, "accepted_attempt": 1, "attempt_source": "remote-path"})
        )
        resources.append(
            dict(
                case=case,
                status="complete",
                registers_per_thread=200,
                local_bytes_per_thread=632 if warps == 4 else 0,
                static_sass_local_counts=dict(loads=0, stores=0),
            )
        )
    path = tmp_path / "resources.json"
    path.write_text(json.dumps(dict(role="control", rows=resources)))
    return tmp_path, path


def test_control_only_matched_audit(diagnostics):
    report = audit(*diagnostics)
    assert report["complete"] and report["count"] == 2
    assert report["eligible_for_fit"] is False
    assert report["matched_warp_pairs"][0]["latency_ratio_low_over_high"] == 2
    assert report["matched_warp_pairs"][0]["local_bytes_per_thread"] == [632, 0]


@pytest.mark.parametrize(
    "field,value",
    [
        ("role", "holdout"),
        ("eligible_for_fit", True),
        ("dropped_records", 1),
        ("median_us", 0),
    ],
)
def test_invalid_diagnostic_rejected(diagnostics, field, value):
    root, _ = diagnostics
    path = root / "controls" / "control_w4.json"
    row = json.loads(path.read_text())
    row[field] = value
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        audit(*diagnostics)


def test_retained_attempt_cannot_be_substituted(diagnostics):
    root, _ = diagnostics
    path = root / "attempts" / "control_w4" / "1.json"
    row = json.loads(path.read_text())
    row["source"]["num_stages"] = 2
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="retained attempt"):
        audit(*diagnostics)


def test_missing_control_retained(diagnostics):
    root, _ = diagnostics
    (root / "controls" / "control_w4.json").unlink()
    report = audit(*diagnostics)
    assert report["count"] == 2 and not report["complete"]
    assert report["rows"][0]["status"] == "missing"
    assert not report["matched_warp_pairs"]
