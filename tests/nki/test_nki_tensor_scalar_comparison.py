"""Float comparison/scale chains preserve a dynamic signed gain exactly."""
import numpy as np
import pytest
import nki.isa as nisa
import nki.language as nl
import triton_viz
from triton_viz.clients import Tracer
from triton_viz.core.trace import launches
from triton_viz.tools.nki_trace_dump import records_to_events

pytestmark=pytest.mark.nki


def test_comparison_boundary_and_signed_gain_have_explicit_memory():
    def kernel(values,out):
        data=nl.ndarray((128,1),nl.float32,buffer=nl.sbuf)
        gain=nl.ndarray((128,1),nl.float32,buffer=nl.sbuf)
        nisa.dma_copy(dst=data,src=values)
        nisa.tensor_scalar(dst=gain,data=data,op0=nl.greater_equal,operand0=3.5,
                           op1=nl.multiply,operand1=2.,engine=nisa.engine.vector)
        nisa.tensor_scalar(dst=gain,data=gain,op0=nl.subtract,operand0=1.,
                           op1=nl.multiply,operand1=1.01,engine=nisa.engine.vector)
        nisa.dma_copy(dst=out,src=gain)
    boundary=np.float32(3.5)
    values=np.resize(np.array([np.nextafter(boundary,np.float32(0)),boundary,
                              np.nextafter(boundary,np.float32(4))],np.float32),(128,1))
    output=np.empty_like(values);triton_viz.clear()
    triton_viz.trace(client=Tracer(),frontend='nki_beta2')(kernel)[(1,)](values,output)
    expected=np.where(values>=boundary,np.float32(1.01),np.float32(-1.01))
    np.testing.assert_array_equal(output,expected)
    events=[e for e in records_to_events(launches[-1].records) if e.get('op')=='compute']
    assert len(events)==2
    assert all(e['input_memories']==['sbuf'] and e['output_memory']=='sbuf' for e in events)
