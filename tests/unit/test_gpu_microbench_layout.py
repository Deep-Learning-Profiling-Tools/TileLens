import json
from pathlib import Path

import pytest

from microbench.gpu.common import cases as declarations
from triton_viz.tools.gpu_tilebench_evaluate import cases as formal_cases


def test_control_declaration_does_not_read_holdouts(monkeypatch):
    original = Path.read_text

    def guarded(path, *args, **kwargs):
        assert "holdout" not in path.name
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    assert len(declarations.load_cases("pilot", "control")) == 36
    assert len(declarations.load_cases("compositional", "control")) == 60
    with pytest.raises(ValueError, match="suite"):
        declarations.load_cases("tilebench254", "control")


def test_holdout_counts_and_shared_nki_declaration():
    assert len(declarations.load_cases("pilot", "holdout")) == 15
    assert len(declarations.load_cases("compositional", "holdout")) == 12
    expected = Path("microbench/inf2_nki/configs/formal_holdouts.json").resolve()
    assert declarations.formal_holdout_splits() == expected
    assert len(formal_cases(expected)) == 254


def test_microbench_entrypoint_is_cpu_safe(capsys):
    from microbench.gpu.harness.run_microbench import main

    assert main(["--root", "unused", "--role", "control", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["stage"] == "collect"


def test_source_identity_includes_relocated_code_and_configs(monkeypatch):
    from triton_viz.tools.gpu_cost_model_pipeline import _source_digest

    baseline = _source_digest()
    original = Path.read_bytes
    for filename in ("pilot_control.json", "measure.py", "kernels.py"):

        def changed(path, target=filename):
            data = original(path)
            return (
                data + b"changed"
                if "gpu" in path.parts and path.name == target
                else data
            )

        with monkeypatch.context() as patch:
            patch.setattr(Path, "read_bytes", changed)
            assert _source_digest() != baseline
