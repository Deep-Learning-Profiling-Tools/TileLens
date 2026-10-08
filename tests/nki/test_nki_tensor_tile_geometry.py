"""Behavioral checks for the opt-in, geometry-dependent Tensor response."""
import copy

import pytest

from triton_viz.tools.nki_tensor_tile_geometry import TensorTileGeometryCalibration
from triton_viz.tools.nki_cost_model import CostModel, TensorCalibrationSurface, simulate

pytestmark = pytest.mark.nki


def dot(m=128, k=128, n=512, seq=0):
    return dict(op="dot", engine="tensor", seq=seq, input_shape=[m, k],
                other_shape=[k, n], output_shape=[m, n],
                flops=2*m*k*n, input_dtypes=["float32", "float32"])


def calibration():
    return TensorTileGeometryCalibration({"float32": (7., 100., 20., 30., 40.)})


def test_equal_flops_different_operand_geometry_is_not_erased():
    a, b = dot(), dot(m=512, n=128, seq=1)
    assert a["flops"] == b["flops"]
    ordinary = TensorCalibrationSurface({"float32": (1000., 0.)},
                                       {"float32": (1., 1e12)})
    result = simulate([a, b], CostModel(tensor_calibration=ordinary,
                                      tensor_tile_geometry_calibration=calibration()))
    durations = [t.end-t.start for t in result.timeline["tensor"]]
    assert durations == pytest.approx([190., 670.])
    assert result.components_ns["tensor_tile_geometry_enabled"] == 1.
    assert result.components_ns["tensor_tile_geometry_dot_units"] == 5.
    assert result.components_ns["tensor_tile_geometry_transport_validated"] == 0.
    assert "tensor_tile_geometry" not in a  # Simulation must not poison later modes.


def test_reference_view_reuse_counts_and_input_versions():
    events = [dict(dot(seq=i), input_storages=[1, 2], input_ranges=[[0, 16], [0, 32]],
                   input_versions=[0, 0], output_storage=3, output_range=[0, 32])
              for i in range(2)]
    c = calibration()
    assert c.assign_events(events)["work_ns"] == 2*100+20+30+40
    events[1]["input_versions"] = [1, 0]
    assert c.assign_events(events)["work_ns"] == 2*100+2*20+30+40


def test_unresolved_views_never_imply_free_reuse():
    c = calibration()
    events = [dot(seq=0), dot(seq=1)]
    assert c.assign_events(events)["work_ns"] == 2*190
    events[0]["input_ranges"] = [[None, 1]]
    assert c.assign_events(events)["work_ns"] == 2*190


def test_stale_override_replaced_and_startup_once_per_dtype():
    events = [dict(dot(seq=i), scheduler_duration_override_ns=999.) for i in range(2)]
    c = calibration()
    assert c.assign_events(events)["startup_ns"] == 7.
    result = simulate(events, CostModel(tensor_tile_geometry_calibration=c))
    assert [t.end-t.start for t in result.timeline["tensor"]] == pytest.approx([190., 190.])
    assert result.timeline["tensor"][0].start == 7.


@pytest.mark.parametrize("change", [
    {"input_shape": [128, 64]}, {"input_shape": [0, 128]},
    {"input_dtypes": []}, {"input_dtypes": ["float32", "bfloat16"]},
    {"input_dtypes": ["float16", "float16"]},
])
def test_unsupported_geometry_fails_explicitly(change):
    with pytest.raises(ValueError):
        calibration().assign_events([dict(dot(), **change)])


def test_default_mode_unchanged_after_geometry_mode():
    events = [dot(), dot(m=512, n=128, seq=1)]
    ordinary = CostModel(tensor_calibration=TensorCalibrationSurface(
        {"float32": (1000., 0.)}, {"float32": (1., 1e12)}))
    before = simulate(copy.deepcopy(events), ordinary).as_dict()
    simulate(events, CostModel(tensor_tile_geometry_calibration=calibration()))
    after = simulate(events, ordinary).as_dict()
    assert before == after
