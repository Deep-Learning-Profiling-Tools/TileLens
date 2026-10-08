"""Shared-resource constraints can change a companion rewrite's marginal value."""
import copy
import pytest
from triton_viz.tools.nki_cost_model import CostModel, simulate
from triton_viz.tools.nki_execution_resources import MemoryExecutionResource, annotate_memory_resources

pytestmark = pytest.mark.nki


def transfer(seq, src, dst, source_memory, destination_memory, duration=0.):
    return dict(op='transfer', engine='vector', seq=seq, src_storage=src,
                dst_storage=dst, src_ptr=src, dst_ptr=dst, src_range=[0,128],
                dst_range=[0,128], mem_src=source_memory, mem_dst=destination_memory,
                bytes=128, scheduler_duration_override_ns=duration)


def compute(seq, storage, engine, duration=40.):
    return dict(op='compute', engine=engine, seq=seq, input_storages=[storage],
                input_ranges=[[0,128]], output_storage=storage,
                output_range=[0,128], scheduler_duration_override_ns=duration)


def program(label):
    setup=[transfer(0,10,1,'sbuf','psum'),transfer(1,20,2,'sbuf','psum')]
    vector=compute(2,1,'vector')
    scalar_psum=compute(3,2,'scalar')
    scalar_sbuf=compute(3,3,'scalar')
    copy_out=transfer(4,2,3,'psum','sbuf',2.)
    bodies={'S':[vector,scalar_psum,copy_out],
            'A':[vector,copy_out,scalar_sbuf],
            'B':[scalar_psum,copy_out,vector],
            'AB':[copy_out,scalar_sbuf,vector]}
    events=copy.deepcopy(setup+bodies[label])
    annotate_memory_resources(events,[MemoryExecutionResource('candidate_psum_port',('vector','scalar'),('psum',))])
    return events


def test_location_and_order_are_both_required_for_the_modeled_gain():
    model=CostModel(cross_engine_sync_ns=0.,exclusive_execution_resources=('candidate_psum_port',))
    predictions={c:simulate(program(c),model).predicted_latency_ns for c in ['S','A','B','AB']}
    assert predictions=={'S':82.,'A':82.,'B':82.,'AB':42.}
    independent={c:simulate(program(c),CostModel(cross_engine_sync_ns=0.)).predicted_latency_ns for c in predictions}
    assert independent=={'S':42.,'A':82.,'B':82.,'AB':42.}
    # Same per-engine work; the information is a conditional overlap constraint.
    assert simulate(program('S'),model).engine_busy_ns==simulate(program('AB'),model).engine_busy_ns


def test_unrelated_storage_is_still_subject_to_the_shared_resource():
    events=program('S');model=CostModel(exclusive_execution_resources=('candidate_psum_port',),cross_engine_sync_ns=0.)
    result=simulate(events,model)
    assert result.timeline['scalar'][0].start==40.
    assert result.components_ns['exclusive_resource_wait_ns']==40.


def test_missing_or_conflicting_provenance_cannot_become_free_overlap():
    group=MemoryExecutionResource('p',('vector','scalar'),('psum',))
    with pytest.raises(ValueError,match='provenance missing'):
        annotate_memory_resources([compute(0,99,'scalar')],[group])
    with pytest.raises(ValueError,match='conflicting'):
        annotate_memory_resources([transfer(0,1,2,'psum','sbuf'),transfer(1,1,3,'sbuf','sbuf')],[group])


def test_opt_in_requires_an_explicit_binding_for_every_event():
    with pytest.raises(ValueError,match='annotation'):
        simulate([compute(0,1,'scalar')],CostModel(exclusive_execution_resources=('p',)))
    event=dict(compute(0,1,'scalar'),execution_resources=['unknown'])
    with pytest.raises(ValueError,match='annotation'):
        simulate([event],CostModel(exclusive_execution_resources=('p',)))
