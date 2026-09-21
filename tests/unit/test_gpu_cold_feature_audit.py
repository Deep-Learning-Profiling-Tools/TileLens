import pytest

from triton_viz.tools.gpu_cold_feature_audit import collision_bound


def test_equal_vectors_bound_includes_every_case():
    rows = [
        dict(id=str(i), latency_us=t, features=dict(work=1, warps=i + 1))
        for i, t in enumerate((1, 10, 100))
    ]
    report = collision_bound(rows, ("work",))
    assert report["oracle_mape_floor_pct"] == pytest.approx(63)
    assert report["collisions"][0]["latency_ratio"] == 100
    assert report["count"] == 3
    split = collision_bound(rows, ("work", "warps"))
    assert split["oracle_mape_floor_pct"] == 0
    assert split["distinct_vectors"] == 3  # Not a generalization claim.


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_bad_measurements_are_rejected_not_filtered(value):
    with pytest.raises(ValueError):
        collision_bound(
            [dict(id="bad", latency_us=value, features=dict(work=1))], ("work",)
        )


def test_incomplete_controls_stop_before_feature_or_label_access(monkeypatch, tmp_path):
    from triton_viz.tools import gpu_cold_feature_audit as module

    monkeypatch.setattr(
        module, "audit", lambda root: dict(measurement_integrity_passed=False)
    )
    monkeypatch.setattr(
        module, "_read", lambda path: pytest.fail("Read after failed integrity audit")
    )
    with pytest.raises(ValueError, match="Incomplete"):
        module.run([tmp_path], 48)


def test_protocol_mixing_is_rejected_before_second_collection_rows(
    monkeypatch, tmp_path
):
    from triton_viz.tools import gpu_cold_feature_audit as module

    monkeypatch.setattr(
        module, "audit", lambda root: dict(measurement_integrity_passed=True)
    )
    monkeypatch.setattr(
        module,
        "_read",
        lambda path: dict(
            metric=path.parent.name,
            timestamp_method="software_serial",
            launch_mode="individual",
            library_sha256="test",
            kernels_per_sample=32,
            packages={},
            cases=[],
        ),
    )
    with pytest.raises(ValueError, match="mix measurement"):
        module.run([tmp_path / "first", tmp_path / "second"], 48)
