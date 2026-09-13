# NKI 254 logical cases on Triton/GPU: frozen-model diagnostic

Date: 2026-09-11. GPU: NVIDIA GB10, 48 SMs. No calibration coefficient or feature
selection was changed using these targets.

## Outcome

All 254 logical cases now have a prediction and hardware measurement. The
selected source-dependency model has **18.5475% MAPE**; the aggregate baseline,
fitted to the same previous 60 controls, has **21.4375% MAPE**. Both are evaluated
against the same hardware results. This is a 2.8900 percentage-point reduction,
approximately 13.48% relative.

**All 254 are OOD. These numbers are diagnostic extrapolations, not accepted
in-domain accuracy.** Strict public prediction rejects these configurations.
Some OOD reasons also identify operations with no price, so the diagnostic
predictions may omit costs, not merely extrapolate calibrated coefficients.

| Operator | Cases | Selected model MAPE |
| --- | ---: | ---: |
| interleave | 30 | 16.33% |
| kl_divergence | 30 | 30.55% |
| layernorm | 30 | 28.20% |
| mul2 | 30 | 6.94% |
| relu | 30 | 5.12% |
| rmsnorm | 30 | 21.80% |
| sigmoid | 30 | 7.61% |
| softmax | 30 | 21.35% |
| matmul | 10 | 21.94% |
| attention, explicit semantic adapter | 4 | 88.74% |

By dtype: FP32 129 cases, 21.25%; BF16 125 cases, 15.76%. MAPE is the mean over
individual points, not an unweighted mean over operator-family averages.

## Matching the NKI suite

The case list is the deduplicated union in
`microbench/inf2_nki/configs/formal_holdouts.json`: 240 vector/reduction cases,
10 square-output matmuls, and four attention cases. Overlapping formal/full/
auxiliary splits must not be counted twice.

- Vector/reduction: eight operators × three row counts (1, 16, 128) × five
  column counts (128, 512, 1024, 2048, 4096) × two dtypes.
- Matmul: `rows=M=N=512`, `cols=K` in 512/1024/2048/4096/8192, FP32 and BF16.
- Attention: batch=heads=1, Q/K sequence lengths 128, Q/K dimension 128,
  V dimension 64/128/256/512, noncausal FP32 attention scaled by `1/sqrt(128)`.

This matches logical operators, shapes and storage dtypes, not physical tiles,
compiler instructions, or identical floating-point rounding across platforms.
Input seed is zero using a CPU Torch generator; this is not a claim to reproduce
the NKI NumPy input bytes. Inputs, constants, and references are deterministic.

245 cases use unmodified Tilebench Triton kernels with fixed default launch
configurations and autotuning disabled. The local Tilebench revision cannot
directly execute nine remaining cases:

1. Five BF16 matmuls: the default table omits BF16. The explicit adapter uses
   the existing FP16 tile configuration (128×128×64, eight warps, three stages)
   for BF16 inputs/output. The kernel body remains unchanged, including its
   FP16 intermediate epilogue conversion before BF16 storage.
2. Four attention cases: NKI's source is `examples/nki_beta2/tiled_attention.py`,
   not a Tilebench `tiled_attention` directory. Tilebench FlashAttention fixes
   the V/output dimension to the Q dimension and cannot directly cover the
   family. `microbench/gpu/holdouts/attention.py` is a separately labeled semantic adapter
   with query tile 32, key tile 32, value tile 64 and IEEE FP32 dots. It is not
   an autotuned FlashAttention performance result.

Thus the full result is **245 original implementations + nine explicit adapters**,
not “254 unmodified Tilebench implementations.” In particular, do not compare
18.55% directly with NKI's 8.83% as if timing protocols and precision semantics
were identical. GPU timing here is CUDA-graph steady-cache latency; NKI uses its
separate NC-p50 and compilation-trial protocol.

## Collection, repairs, and data integrity

The first run attempted every declared point. It collected 240 accepted hardware
measurements, but 15 BF16 mul2 observations failed due to a missing interpreter
scalar constructor. Five FP32 matmuls failed an inappropriate FP32 output
tolerance because their source explicitly requests TF32. Five BF16 matmuls and
four attentions needed the adapters above.

The supplement attempted precisely these 29 cases. FP32 matmul validation now
uses a forward error bound for TF32 input quantization and FP32 accumulation:
`(2*u + u*u + gamma_K) * (abs(A) @ abs(B))`, with `u=2^-10` and
`gamma_K=K*2^-24/(1-K*2^-24)`. Its measured relative L2 output error is about
0.00077–0.00079. This changes correctness validation, not source precision or
latency-model fitting. Other GPU outputs passed the recorded elementwise checks.

Merging always keeps the **first accepted hardware measurement**. For BF16 mul2,
the supplement supplies the missing prediction but the first run's hardware
latency remains authoritative. No selection is made based on prediction error.
Original failures, all retries, and supplemental measurements remain available.

A subsequent CPU numerical audit found Triton interpreter BF16 arithmetic used
the underlying uint16 bit patterns. `triton_observe.py` now scopes compatibility
wrappers to observation: construct round-to-nearest-even BF16 constants, decode
BF16 arithmetic/dot operands, and restore original methods afterwards. GPU
compilation and kernels are unchanged. After this repair, **all 254 CPU outputs
passed their references**, and **all 254 predicted latencies were unchanged**.
The targets have fixed, shape-dependent control flow, explaining why the previous
bad arithmetic did not change the extracted structural work in this experiment.
The corrected observations are archived separately; do not reuse the earlier
invalid BF16 numerical observations as a reference implementation.

Shared-machine timing used the existing NVML monitor and explicitly allowed
pre-existing idle desktop graphics processes. The primary run retained 247 timing
attempts, seven rejected; the supplement retained 29 attempts, none rejected.
No other processes or hardware settings were changed. This monitor does not
establish exclusivity, and the experiment has no independent-session confidence
interval. Compilation/warmup are excluded; this is not cold-cache latency.

## Why the model is OOD, and what this suggests

All cases have an uncovered launch/tile/dtype configuration relative to the small
60-control calibration. Additional reasons include unpriced `join` layouts,
`ternary_op`, comparisons, integer division, and work beyond feature ranges.
The artifact retains each reason rather than weakening the gate.

The observed errors suggest the following hypotheses for **new controls**, not
constants to fit against these 254 targets:

- Small-program long loops need loop-carried dependencies and local latency
  controls; flattening source repetitions can lose serial structure. KL and
  normalization errors are particularly large at small row counts/large columns.
- Dot input precision and actual lowering must distinguish IEEE FP32, TF32,
  BF16 and FP16. Current tensor pricing primarily comes from small FP16 controls;
  treating the IEEE attention adapter's dot work similarly is inadequate.
- Layout, predication, tile/warp configuration, and working-set regimes need
  their own coverage and control-only lowering evidence.

These 254 cases are now observed diagnostics. Future model development should
declare another untouched validation suite, or label repeated results on these
points as development-set results rather than fresh held-out evidence.

## Artifacts and commands

Artifacts live in ignored `downloads/gpu_tilebench254_20260911/`:

- `primary/run/`, `supplement/run/`: raw attempts, observations, failures,
  measurements and source fingerprints.
- `merged/cases.csv`, `merged/report.json`, `merged/baseline.json`: 254-point
  report and same-measurement aggregate comparison.
- `cpu_audit/audited/`: corrected observations, CPU reference passes, and all
  prediction deltas (zero).
- `primary.tar.gz`, `supplement.tar.gz`, `audited.tar.gz`: archived data/code.

Use the previously frozen model from the 60-control experiment; do not run fit:

```bash
python -m triton_viz.tools.gpu_tilebench_evaluate --root PRIMARY \
  --tilebench-dir TILEBENCH/benchmarks/operators \
  --splits microbench/inf2_nki/configs/formal_holdouts.json \
  --calibration FROZEN_MODEL --calibration-manifest CONTROL_MANIFEST
python -m triton_viz.tools.gpu_tilebench_evaluate --root SUPPLEMENT \
  --tilebench-dir TILEBENCH/benchmarks/operators \
  --splits microbench/inf2_nki/configs/formal_holdouts.json \
  --calibration FROZEN_MODEL --calibration-manifest CONTROL_MANIFEST --supplement-only
python -m triton_viz.tools.gpu_tilebench_report --primary PRIMARY \
  --supplement SUPPLEMENT --output MERGED --baseline-ablation FROZEN_ABLATION
python -m triton_viz.tools.gpu_tilebench_observe_audit --merged MERGED \
  --tilebench-dir TILEBENCH/benchmarks/operators --model FROZEN_MODEL --output AUDITED
```

The collection evaluator uses the shared-desktop allowance described above.
Use reserved hardware for publication-quality replication. A new source snapshot
changes the measurement identity; use fresh run directories. The frozen model
fingerprint remains separately recorded, with runtime/hardware compatibility
checked rather than pretending that adding an evaluator is a new calibration.
