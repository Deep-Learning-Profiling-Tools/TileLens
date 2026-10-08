"""A compute-only constant must not need a fake transfer to obtain its type."""
import numpy as np
import pytest
import triton_viz
import nki.isa as nisa
import nki.language as nl
import neuronxcc.nki.language as legacy_nl
from triton_viz.clients import Tracer
from triton_viz.core.trace import launches
from triton_viz.tools.nki_trace_dump import records_to_events

pytestmark=pytest.mark.nki


@pytest.mark.parametrize('buffer_name',['sbuf','psum'])
def test_compute_memory_includes_constant_and_dynamic_scale_operands(buffer_name):
    def kernel(out):
        data=nl.ndarray((128,8),nl.float32,buffer=getattr(nl,buffer_name))
        scale=nl.ndarray((128,1),nl.float32,buffer=nl.sbuf)
        bias=nl.ndarray((128,1),nl.float32,buffer=nl.sbuf)
        nisa.memset(dst=data,value=2.)
        nisa.memset(dst=scale,value=1.5)
        nisa.memset(dst=bias,value=0.)
        nisa.activation(dst=data,data=data,op=nl.relu,scale=scale,bias=bias)
        copy=nl.ndarray((128,8),nl.float32,buffer=nl.sbuf)
        nisa.tensor_copy(dst=copy,src=data)
        nisa.dma_copy(dst=out,src=copy)
    triton_viz.clear()
    output=np.empty((128,8),np.float32)
    triton_viz.trace(client=Tracer(),frontend='nki_beta2')(kernel)[(1,)](output)
    np.testing.assert_allclose(output,3.)
    events=records_to_events(launches[-1].records)
    activation=next(e for e in events if e.get('api_op')=='relu')
    assert activation['input_memories']==[buffer_name,'sbuf','sbuf']
    assert activation['output_memory']==buffer_name
    constants=[e for e in events if e.get('api_op')=='memset']
    assert constants and all(e['output_memory'] in {'sbuf','psum'} for e in constants)


def test_unknown_memory_is_not_assumed_sbuf():
    from triton_viz.core.simulation.nki import NDArray
    def kernel(value):
        return legacy_nl.exp(value)
    triton_viz.clear()
    traced=triton_viz.trace(client=Tracer(),frontend='nki')(kernel)
    traced[(1,)](NDArray(value=np.ones((128,8),np.float32)))
    event=next(e for e in records_to_events(launches[-1].records) if e.get('op')=='compute')
    assert event['input_memories']==[None]
    assert event['output_memory'] is None
