# Source-semantic dot precision calibration

The 480-control model replays the historical 254 measurements at 16.2939% MAPE,
but matmul MAPE is 107.7620%. At K=8192 the old global tensor term contributes
65.52 us and the per-program tensor term contributes 282.34 us to both FP32 and
BF16 predictions. The shared per-program price is therefore the larger issue.
Both prices mix dot input precision and extrapolate beyond the original short
control loops. The BF16 epilogue's FP16 cast does not determine its dot input.

## Observation and model

Dot events now retain both operand dtypes, accumulator dtype and normalized
`input_precision`, directly from interpreter arguments before BF16 compatibility
conversion. No target compilation or compiler metrics are used as features.

Four classes are supported: IEEE FP32, TF32 with FP32 inputs, BF16 and FP16
(FP32 accumulators). Unknown/mixed inputs and other modes such as TF32x3 are
explicitly OOD. Legacy traces lacking precision metadata require reobservation.

`--dot-precision` opts into two control-selected candidates:

- `dot_precision_aggregate`: replace the global mixed tensor price with four
  precision-specific global FLOP terms.
- `dot_precision_combined`: replace both the global and per-program mixed
  tensor terms with four prices each; retain existing non-tensor terms.

The four per-program p90 statistics include zero-work programs. Legacy feature
sets and their coefficients remain unchanged. Precision configurations are
checked alongside launch/tile/dtype coverage and feature ranges at prediction.

## New controls and experiment

The `precision` suite includes all 480 coverage controls plus 32 new dot controls:
four input/precision classes, 2/8/32/192 programs, and 17/65 iterations of a
128x128x64 dot at eight warps and one stage. CV groups remain program counts;
no target name, target measurement or epilogue label determines a price.

All 512 controls are collected in a new run with the new metadata. The numerical
reference and constants are unchanged from coverage controls. Fitting uses only
control paths, with a runtime read guard; both ordinary and nested CV must meet
the existing 20% gate. Rejected candidates are not used for prediction.

```bash
python -m triton_viz.tools.gpu_cost_model_pipeline collect \
  --root NEW_RUN --suite precision --role control --allow-idle-graphics
python -m triton_viz.tools.gpu_distribution_experiments fit \
  --root NEW_RUN --output NEW_FIT --dot-precision
```

The final 254-point comparison reobserves CPU source with the frozen new model
and retains the original accepted hardware measurements. It is development-set
replay, not a new independent holdout result. Source observations precede target
compilation; CPU replay prohibits compilation and CUDA initialization entirely.

## Completed experiment

All 512 controls were collected on GB10 under the existing monitored shared
desktop protocol. The runtime fit-read audit records exactly 513 unique JSON
reads: one control manifest and 512 control rows, no validation or target files.

`dot_precision_combined` passes ordinary and nested control CV at **19.9038556%**,
just below the unchanged 20% gate. `dot_precision_aggregate` fails at
**47.4709482%** and is not used for target prediction. Some individual folds
exceed 20%; the existing gate is the aggregate CV MAPE, not an every-fold gate.

All 254 source observations passed CPU numerical checks with target compilation
and CUDA initialization disabled. Evaluation retains every original hardware
measurement; no case is deleted or retimed. The new result is **15.6162942% MAPE**,
with **185/254 OOD**, against the prior 480-control replay's **16.2939153%**.
This is a 0.6776211 percentage-point reduction. It combines new control data,
precision features and corrected source observations; it is not an isolated
causal estimate of the precision split or an independent-session confidence claim.

| Group | Previous 480-control replay | Precision model |
| --- | ---: | ---: |
| All 254 | 16.2939% | 15.6163% |
| FP32, 129 cases | 16.2506% | 17.8209% |
| BF16, 125 cases | 16.3386% | 13.3411% |
| All matmul, 10 cases | 107.7620% | 99.4100% |
| FP32/TF32 matmul, 5 cases | 47.9556% | 103.7181% |
| BF16 matmul, 5 cases | 167.5683% | 95.1019% |
| Attention adapter, 4 cases | 81.6171% | 65.5770% |

The matmul problem is **not solved**: BF16 improves, TF32 regresses, and all ten
matmuls remain OOD. The implementation no longer mixes input precision, but
precision alone does not explain tile scheduling, reuse, pipeline depth and
local throughput. No further candidate is selected using these target errors.

Learned prices (microseconds per source FLOP):

| Dot input class | Global term | Per-program p90 term |
| --- | ---: | ---: |
| IEEE FP32 | 6.83310e-8 | 3.03754e-6 |
| TF32 | 3.96688e-8 | 1.25307e-6 |
| BF16 | 1.70375e-8 | 6.44899e-7 |
| FP16 | 1.69767e-8 | 6.40846e-7 |

At FP32 matmul K=8192, prediction is 560.79 us versus measured 262.12 us:
170.38 us global dot work + 336.37 us per-program dot work + 53.24 us memory
+ 0.80 us launch. BF16 predicts 273.71 us versus 131.38 us. Thus the dominant
local dot cost still transfers poorly from isolated dot controls to tiled matmul.
This motivates separate control experiments for tile geometry, shared operand
reuse and pipeline scheduling; it does not justify a target correction factor.

Recorded BF16 matmul operands are `bf16/bf16` with FP32 accumulation. Its source
passes the `tf32` precision flag; the classifier still prices BF16, not FP32/TF32.
That exact flag/configuration was absent from the controls and remains explicitly
OOD, alongside unpriced integer/ternary operations and uncovered launch geometry.

Local artifacts: `downloads/gpu_precision_20260916/` contains `fit/`, the control
manifest, all replay rows and observations under `replay/`, `results.tar.gz`, and
`controls.tar.gz` (accepted rows plus all measurement attempts). The guarded fit
and CPU replay scripts are included in `results.tar.gz`. Frozen model digest:
`9c0f9320edf2b464060b`. Remote source/data are retained under
`~/tmp/gpu_precision_20260916/`; the source archive is
`~/tmp/gpu_precision_20260916_src.tar.gz`.

Validation: 63 focused tests pass, covering four-class metadata/work accounting,
legacy-metadata rejection, control-only fitting, public prediction, paired
controls, unchanged legacy distribution behavior and model digests. Ruff and
diff checks pass. Remote four-precision CPU tests passed before collection.
