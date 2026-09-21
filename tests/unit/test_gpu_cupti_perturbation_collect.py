import sys

import pytest

from triton_viz.tools import gpu_cupti_perturbation_collect as collect


def snapshot(processes=()):
    return dict(
        processes=[dict(pid=p) for p in processes],
        graphics_processes=[],
        utilization_pct=0,
        uuid="GPU-test",
        driver="test",
        index=0,
    )


def test_declared_matrix_has_balanced_order_without_adaptive_trials():
    cases = collect.declared_trials()
    assert len(cases) == len({c["id"] for c in cases}) == 48
    for i in range(0, 48, 2):
        a, b = cases[i : i + 2]
        assert {a["mode"], b["mode"]} == {"none", "software_serial"}
        assert all(a[k] == b[k] for k in ("programs", "iterations", "trial"))
        assert a["mode"] == ("none" if a["trial"] % 2 == 0 else "software_serial")


def test_new_workloads_have_disjoint_explicit_trial_identities():
    groups = [collect.declared_trials(kind) for kind in ("fma", "tensor", "local")]
    assert len({case["id"] for group in groups for case in group}) == 144
    for kind, cases in zip(("tensor", "local"), groups[1:]):
        assert all(case["workload"] == kind for case in cases)


def test_foreign_baseline_prevents_child_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "snapshot", lambda _: snapshot([1234]))
    monkeypatch.setattr(
        collect.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch")
    )
    with pytest.raises(RuntimeError, match="Foreign"):
        collect.monitored_process(["unused"], tmp_path / "raw.log")
    assert not (tmp_path / "raw.log").exists()


def test_monitored_cpu_child_retains_output(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "snapshot", lambda _: snapshot())
    result = collect.monitored_process(
        [sys.executable, "-c", "print('diagnostic')"], tmp_path / "raw.log"
    )
    assert result["returncode"] == 0 and not result["contaminated"]
    assert len(result["samples"]) >= 3
    assert "diagnostic" in (tmp_path / "raw.log").read_text()


def test_foreign_arrival_only_terminates_our_child(tmp_path, monkeypatch):
    calls = 0

    def sample(_):
        nonlocal calls
        calls += 1
        return snapshot([] if calls == 1 else [987654])

    monkeypatch.setattr(collect, "snapshot", sample)
    result = collect.monitored_process(
        [sys.executable, "-c", "import time; time.sleep(30)"], tmp_path / "raw.log"
    )
    assert result["contaminated"] and result["returncode"] != 0
    assert result["child_pid"] != 987654
    assert result["elapsed_seconds"] < 10
