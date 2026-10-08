"""Readiness and forwarding behaviors, independent of any target performance."""
import pytest
from triton_viz.tools.nki_tensor_initiation import TensorInitiationCalibration
from triton_viz.tools.nki_cost_model import CostModel, simulate

pytestmark = pytest.mark.nki


def calibration():
    return TensorInitiationCalibration(2., {'float32': (0., 300.)})


def dot(seq=0, accumulate=False):
    return dict(op='dot', engine='tensor', seq=seq, input_shape=[64,128], other_shape=[128,128],
        output_shape=[64,128], input_dtypes=['float32','float32'], flops=2*64*128*128,
        input_storages=[1,2], input_ranges=[[0,128],[0,128]],
        output_storage=3, output_range=[0,128], tensor_pipeline_accumulate=accumulate)


def test_operand_swap_changes_initiation_without_flop_change():
    a=dot(); b=dict(a,input_shape=[128,128],other_shape=[128,64])
    assert calibration().timing(a)['initiation_ns']==256.
    assert calibration().timing(b)['initiation_ns']==128.
    assert calibration().timing(a)['completion_ns']==556.


def test_accumulation_forwards_but_consumer_waits_for_result():
    consumer=dict(op='transfer',engine='vector',seq=2,bytes=128,
                  input_storages=[3],input_ranges=[[0,128]],
                  output_storage=4,output_range=[0,128])
    model=CostModel(tensor_initiation_calibration=calibration(),cross_engine_sync_ns=0.)
    result=simulate([dot(),dot(1,True),consumer],model)
    assert result.timeline['tensor'][1].start==256.
    assert result.timeline['tensor'][1].ready_end==812.
    assert result.timeline['vector'][0].start==812.
    assert result.engine_busy_ns['tensor']==512.
    assert result.predicted_latency_ns>=812.


def test_overwrite_and_missing_accumulation_evidence_wait_for_completion():
    for second in [dot(1), {k:v for k,v in dot(1,True).items() if k!='tensor_pipeline_accumulate'}]:
        result=simulate([dot(),second],CostModel(tensor_initiation_calibration=calibration()))
        assert result.timeline['tensor'][1].start==556.


def test_read_before_overwrite_cannot_be_bypassed_by_forwarding():
    consumer=dict(op='transfer',engine='vector',seq=1,bytes=128,
                  input_storages=[3],input_ranges=[[0,128]],
                  output_storage=4,output_range=[0,128])
    result=simulate([dot(),consumer,dot(2,True)],
                    CostModel(tensor_initiation_calibration=calibration(),cross_engine_sync_ns=0.))
    assert result.timeline['tensor'][1].start>=result.timeline['vector'][0].end


def test_transpose_has_matmul_readiness_instead_of_flop_proxy():
    event=dict(op='tensor_transpose',input_shape=[64,128],input_dtypes=['float32'])
    assert calibration().timing(event)['initiation_ns']==128.
    assert calibration().timing(event)['completion_ns']==428.


def test_no_silent_unsupported_tile_or_unknown_dtype():
    with pytest.raises(ValueError):
        calibration().timing(dict(dot(),input_shape=[129,128]))
    with pytest.raises(ValueError):
        calibration().timing(dict(dot(),input_dtypes=[]))
