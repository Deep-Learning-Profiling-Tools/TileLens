import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from triton_viz.tools import gpu_pressure_probe_batch as batch


@pytest.mark.parametrize("unstable", [False, True])
def test_pressure_batch_retains_declared_matrix_and_stops_on_instability(
    tmp_path, monkeypatch, unstable
):
    called = []

    def run(command, **kwargs):
        assert "--graph-samples" in command
        assert command[command.index("--suite") + 1] == "pressure"
        output = Path(command[command.index("--output") + 1])
        called.append(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                dict(
                    median_us=1,
                    relative_span=0.2 if unstable else 0.1,
                    unstable=unstable,
                    eligible_for_fit=False,
                )
            )
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(batch.subprocess, "run", run)
    output = tmp_path / "batch"
    args = ["--library", str(tmp_path / "bridge.so"), "--output", str(output)]
    if unstable:
        with pytest.raises(RuntimeError, match="unstable"):
            batch.main(args)
    else:
        batch.main(args)
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["cases"]) == 32
    assert not manifest["eligible_for_fit"]
    assert len(called) == (3 if unstable else 32)
    assert all(p.exists() for p in called)
    with pytest.raises(ValueError, match="fresh"):
        batch.main(args)


def test_pressure_batch_accepts_first_stable_not_fastest(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "load_cases", lambda *_: [{"id": "declared"}])
    calls = []

    def run(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        calls.append(output)
        unstable = len(calls) == 1
        output.write_text(
            json.dumps(
                dict(
                    median_us=0.01 if unstable else 10,
                    relative_span=0.2 if unstable else 0.1,
                    unstable=unstable,
                    eligible_for_fit=False,
                )
            )
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(batch.subprocess, "run", run)
    output = tmp_path / "batch"
    batch.main(["--library", str(tmp_path / "bridge.so"), "--output", str(output)])
    accepted = json.loads((output / "controls" / "declared.json").read_text())
    assert accepted["accepted_attempt"] == 2 and accepted["median_us"] == 10
    assert len(calls) == 2 and all(p.exists() for p in calls)


def test_pressure_batch_timeout_retains_three_attempt_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "load_cases", lambda *_: [{"id": "declared"}])

    def run(command, **kwargs):
        raise batch.subprocess.TimeoutExpired(command, 180)

    monkeypatch.setattr(batch.subprocess, "run", run)
    output = tmp_path / "batch"
    with pytest.raises(RuntimeError, match="after 3 batches"):
        batch.main(["--library", str(tmp_path / "bridge.so"), "--output", str(output)])
    logs = list((output / "attempts" / "declared").glob("*.log"))
    assert len(logs) == 3
    assert all("timed out" in log.read_text() for log in logs)
    assert not (output / "controls").exists()


def test_monitored_batch_rejects_missing_monitoring(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "load_cases", lambda *_: [{"id": "declared"}])

    def run(command, **kwargs):
        assert "--monitored" in command and "--capture-cache" in command
        output = Path(command[command.index("--output") + 1])
        output.write_text(
            json.dumps(dict(unstable=False, median_us=1, relative_span=0))
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(batch.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="after 3 batches"):
        batch.main(
            [
                "--library",
                "unused",
                "--output",
                str(tmp_path / "batch"),
                "--monitored",
                "--capture-cache",
            ]
        )


def test_explicit_direct_method_is_declared_before_collection(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "load_cases", lambda *_: [{"id": "declared"}])

    def run(command, **kwargs):
        assert "--graph-samples" not in command
        assert command[command.index("--timestamp-method") + 1] == "software_serial"
        assert command[command.index("--kernels-per-sample") + 1] == "32"
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(batch.subprocess, "run", run)
    root = tmp_path / "batch"
    with pytest.raises(RuntimeError, match="after 3"):
        batch.main(
            [
                "--library",
                "unused",
                "--output",
                str(root),
                "--timestamp-method",
                "software_serial",
                "--launch-mode",
                "individual",
                "--kernels-per-sample",
                "32",
            ]
        )
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["timestamp_method"] == "software_serial"
    assert manifest["launch_mode"] == "individual"
    assert "software_serial" in manifest["metric"]
