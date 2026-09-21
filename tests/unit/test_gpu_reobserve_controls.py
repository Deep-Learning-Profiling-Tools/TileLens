import copy

import pytest

from triton_viz.performance.calibration import stable_digest
from triton_viz.performance.gpu import expand
from triton_viz.tools import gpu_reobserve_controls as tool
from triton_viz.tools.gpu_cost_model_pipeline import _read, _write


@pytest.mark.parametrize("mismatch", [False, True])
def test_control_reobservation_preserves_provenance_and_rejects_event_changes(
    tmp_path, monkeypatch, mismatch
):
    root, output = tmp_path / "parent", tmp_path / "new"
    case = dict(id="one", kind="coverage_dot", programs=8)
    identity = dict(
        suite="precision",
        uuid="test",
        sm_count=48,
        driver="test",
        packages={},
        metric="test",
        capability=[12, 1],
    )
    fingerprint = stable_digest(identity)
    _write(
        root / "manifest.json",
        dict(
            identity=identity,
            fingerprint=fingerprint,
            splits={"control": [case], "holdout": [{"id": "forbidden"}]},
        ),
    )
    source = dict(
        schema="triton-viz.gpu-source.v1",
        program_count=1,
        num_warps=4,
        num_stages=2,
        events=[],
    )
    old = dict(
        role="control",
        case=case,
        cv_group="old",
        fingerprint=fingerprint,
        source=source,
        contaminated=False,
        latency_us=12.5,
        samples_us=[12, 12.5, 13],
        telemetry={"retained": True},
        features=expand(source, sm_count=48)["features"],
    )
    _write(root / "controls/one.json", old)
    monkeypatch.setattr(tool, "load_cases", lambda suite, role: [case])
    monkeypatch.setattr(tool, "_source_digest", lambda: "new-observer")
    monkeypatch.setattr(tool, "prepare", lambda *a: (None, (1,), (), None))
    monkeypatch.setattr(tool, "check_output", lambda *a: None)
    observed = copy.deepcopy(source)
    observed["memory_working_set"] = {"test": "new"}
    if mismatch:
        observed["num_stages"] = 3
    monkeypatch.setattr(tool, "observe", lambda *a, **kw: observed)

    def guarded(path):
        assert "holdouts" not in path.parts
        return _read(path)

    monkeypatch.setattr(tool, "_read", guarded)
    if mismatch:
        with pytest.raises(ValueError, match="Legacy source event mismatch"):
            tool.reobserve([root], output)
        assert not (output / "controls/one.json").exists()
        return
    tool.reobserve([root], output)
    row = _read(output / "controls/one.json")
    assert row["latency_us"] == old["latency_us"]
    assert row["samples_us"] == old["samples_us"]
    assert row["telemetry"] == old["telemetry"]
    assert row["cv_group"] == "8"
    assert row["provenance"]["measurement_digest"] == stable_digest(old)
    assert row["provenance"]["measurement_fingerprint"] == fingerprint
    assert row["fingerprint"] != fingerprint
    assert _read(output / "manifest.json")["splits"]["holdout"] == []
