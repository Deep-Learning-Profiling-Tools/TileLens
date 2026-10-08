# Experimental Tensor initiation and readiness model

Opt in with `CostModel(tensor_initiation_calibration=calibration)`. This mode is
mutually exclusive with the reference-tile geometry response; defaults remain
unchanged. Explicitly pass a Tensor clock and per-dtype kernel startup/readiness
tail, or initialize compound response parameters from independent small controls
using `TensorInitiationCalibration.from_small_controls`.

The [NeuronCore-v2 architecture guide](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/guides/architecture/trainium_inferentia2_arch.html)
describes background stationary loading, faster stationary loads, a 64-cycle MM
initiation floor and a larger FP32 cost. The experimental response approximates
one Dot's initiation demand as the maximum of stationary and moving demands.
Tensor transpose uses its identity-matmul geometry. Single-tile limits are
checked. These guidelines do not establish exact timings for arbitrary kernels.

The scheduler releases the Tensor issue resource at initiation end and publishes
memory results at readiness end. A consumer waits for readiness. An explicitly
marked `tensor_pipeline_accumulate=True` Dot may forward a previous Dot's exact
output interval at initiation end; ordinary overwrites and unknown flags wait
for readiness. This forwarding is a modeling hypothesis about PSUM accumulation,
not a decoded native scheduling guarantee. Existing read-before-write hazards
still constrain reuse. Callers must retain provenance for explicit flags.

The readiness tail fitted to active-union controls is a compound statistical
parameter. It is not an identified instruction latency. Transport from partial
independent controls to other geometries, accumulation, compiler allocation and
mixed pipelines requires prospective validation. Completion transport remains
flagged false. Timeline `end` reports resource initiation end; optional `ready_end`
reports result readiness. Tensor busy sums initiation demand in this mode and
must not be compared directly with profiler ACTIVE. The old aggregate completion
calibration is disabled in this mode because its input semantics differ. Other
engine paths and output hazards remain in the simulator. No accuracy or
agent optimization benefit is established by this API or its behavioral tests.
