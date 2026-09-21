import json
from types import SimpleNamespace

import pytest

from triton_viz.tools import gpu_cache_capacity_collect as collector


def test_all_counter_commands_are_frozen_and_never_change_clocks(tmp_path, monkeypatch):
    commands = []

    def run(command, **kwargs):
        if "--version" in command:
            return SimpleNamespace(stdout="test ncu", returncode=0)
        commands.append(command)
        assert command[command.index("--replay-mode") + 1] == "range"
        assert command[command.index("--clock-control") + 1] == "none"
        assert command[command.index("--cache-control") + 1] == "all"
        assert "duration" not in command[command.index("--metrics") + 1]
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(collector.subprocess, "run", run)
    root = tmp_path / "counters"
    collector.main(["--ncu", "fake", "--output", str(root)])
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["commands"] == commands and len(commands) == 27
    assert manifest["role"] == "control" and not manifest["eligible_for_latency_fit"]
    with pytest.raises(ValueError, match="fresh"):
        collector.main(["--ncu", "fake", "--output", str(root)])


def test_counter_timeout_preserves_manifest_and_failed_log(tmp_path, monkeypatch):
    def run(command, **kwargs):
        if "--version" in command:
            return SimpleNamespace(stdout="test ncu", returncode=0)
        raise collector.subprocess.TimeoutExpired(command, 180)

    monkeypatch.setattr(collector.subprocess, "run", run)
    root = tmp_path / "counters"
    with pytest.raises(RuntimeError, match="timed out"):
        collector.main(["--ncu", "fake", "--output", str(root)])
    assert len(json.loads((root / "manifest.json").read_text())["commands"]) == 27
    assert "no point removed" in (root / "3_none.log").read_text()
