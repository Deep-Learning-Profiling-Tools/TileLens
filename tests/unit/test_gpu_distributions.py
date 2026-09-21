from pathlib import Path

import pytest

from triton_viz.performance.gpu_distributions import distribution_features
from triton_viz.tools import gpu_distribution_experiments as experiments
from triton_viz.tools.gpu_cost_model_pipeline import _write


def test_only_nonconvergence_can_reject_candidate(monkeypatch):
    names = ("aggregate", "program_distribution")

    def fit(rows, features, **kwargs):
        if features == experiments.FEATURE_SETS["aggregate"]:
            raise ValueError("Nonnegative calibration did not converge")
        return {"valid": True}

    monkeypatch.setattr(experiments, "fit_controls", fit)
    models, rejected = experiments.fit_candidates([], names, "test")
    assert list(models) == ["program_distribution"]
    assert rejected["aggregate"]["reason"] == "nonconvergence"
    with pytest.raises(ValueError, match="All candidates failed"):
        experiments.fit_candidates([], names[:1], "test")

    def invalid(*args, **kwargs):
        raise ValueError("Control fingerprint mismatch")

    monkeypatch.setattr(experiments, "fit_controls", invalid)
    with pytest.raises(ValueError, match="Control fingerprint mismatch"):
        experiments.fit_candidates([], names, "test")


def _event(seq, program, op="binary_op", dependencies=(), elements=32):
    return dict(
        seq=seq,
        program=[program, 0, 0],
        op=op,
        dependencies=list(dependencies),
        dtype="fp32",
        primitive="add",
        elements=elements,
        shape=[elements],
        input_shapes=[[elements]],
        sectors=max(1, elements // 8),
    )


def test_dependency_paths_distinguish_chain_from_parallel_equal_work():
    independent = dict(
        program_count=1,
        events=[
            _event(0, 0, "load"),
            _event(1, 0, dependencies=(0,)),
            _event(2, 0, dependencies=(0,)),
        ],
    )
    chained = {
        **independent,
        "events": independent["events"][:2] + [_event(2, 0, dependencies=(1,))],
    }
    parallel, _ = distribution_features(independent)
    serial, _ = distribution_features(chained)
    assert parallel["program_alu_p90"] == serial["program_alu_p90"] == 2
    assert parallel["path_alu_p90"] == 1
    assert serial["path_alu_p90"] == 2


def test_program_tail_differs_from_global_average():
    source = dict(program_count=2, events=[_event(0, 0), _event(1, 1, elements=288)])
    features, distributions = distribution_features(source)
    assert features["program_alu_p90"] == pytest.approx(8.2)
    assert distributions["program_alu"]["mean"] == 5
    assert distributions["program_alu"]["max"] == 9


@pytest.mark.parametrize(
    "source",
    [
        dict(program_count=2, events=[_event(0, 0)]),
        dict(program_count=1, events=[_event(0, 0, dependencies=(5,))]),
        dict(program_count=2, events=[_event(0, 0), _event(1, 1, dependencies=(0,))]),
    ],
)
def test_incomplete_or_unsupported_dependencies_are_rejected(source):
    with pytest.raises(ValueError):
        distribution_features(source)


def test_nested_ablation_reads_controls_only(tmp_path, monkeypatch):
    cases = [{"id": str(n)} for n in (1, 2, 4, 8)]
    _write(
        tmp_path / "manifest.json",
        dict(
            fingerprint="test",
            identity={"sm_count": 48},
            splits={"control": cases, "holdout": [{"id": "must_not_open"}]},
        ),
    )
    for case in cases:
        n = int(case["id"])
        source = dict(
            schema="triton-viz.gpu-source.v1",
            program_count=n,
            num_warps=4,
            num_stages=2,
            events=[_event(p, p, "load", elements=512) for p in range(n)],
        )
        _write(
            tmp_path / "controls" / (case["id"] + ".json"),
            dict(
                role="control",
                contaminated=False,
                fingerprint="test",
                cv_group=case["id"],
                source=source,
                latency_us=0.8 + 0.1 * n,
            ),
        )
    original = experiments._read

    def guarded(path):
        assert "holdouts" not in Path(path).parts
        return original(path)

    monkeypatch.setattr(experiments, "_read", guarded)
    experiments.fit(tmp_path, tmp_path / "ablation")
    artifact = original(tmp_path / "ablation/frozen_ablation.json")
    assert artifact["selected"] == "aggregate"
    assert artifact["nested_mape_pct"] < 1e-5
    for fold in artifact["nested_cv"]:
        assert set(fold["candidate_cv_mape_pct"]) == set(artifact["models"])
        assert fold["selection_margin_pct"] >= 0
    from triton_viz.performance import GpuBackend, predict_latency

    distribution_model = artifact["models"]["program_distribution"]
    result = predict_latency(source, GpuBackend(distribution_model, "test", 48))
    assert result.latency_ns == pytest.approx(1600, abs=1e-4)
    assert result.diagnostics["feature_set"] == "program_distribution"
