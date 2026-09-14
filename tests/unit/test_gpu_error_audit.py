import pytest

from triton_viz.tools.gpu_cost_model_pipeline import _write
from triton_viz.tools.gpu_error_audit import audit


def _run(root, predictions=(1, 4, 6)):
    cases = [
        dict(id=str(i), op="a" if i < 2 else "b", dtype="float32", rows=1)
        for i in range(3)
    ]
    _write(root / "manifest.json", {"cases": cases})
    for case, predicted in zip(cases, predictions):
        _write(
            root / "cases" / (case["id"] + ".json"),
            {
                "case": case,
                "predicted_us": predicted,
                "measured_us": 2,
                "error_pct": 999,  # Audit must recompute rather than trust cached metrics.
                "ood_reasons": ["outside_control_domain:alu:5:[0,1]", "unknown_op:dot"],
            },
        )


def test_weighted_contribution_and_signed_bias(tmp_path):
    _run(tmp_path)
    _write(tmp_path / "cases" / "undeclared.json", {"invalid": True})
    report = audit(tmp_path)
    assert report["overall"]["mape_pct"] == pytest.approx(350 / 3)
    assert report["overall"]["bias_pct"] == pytest.approx(250 / 3)
    assert report["operator_priority"] == ["b", "a"]
    for groups in report["groups"].values():
        assert sum(g["contribution_pp"] for g in groups.values()) == pytest.approx(
            350 / 3
        )
    assert report["overlapping_ood_groups"]["outside_control_domain:alu"]["count"] == 3


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf")])
def test_rejects_invalid_prediction(tmp_path, bad):
    _run(tmp_path, (bad, 4, 6))
    with pytest.raises(ValueError, match="Invalid latency"):
        audit(tmp_path)
