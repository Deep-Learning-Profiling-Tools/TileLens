# Source-structure GPU model: Concorde-inspired experiment

Date: 2026-09-11. This is an experimental extension, not a reproduction of
Concorde. The shared Observe → Expand → Price → Combine API is unchanged.

## Result

On NVIDIA GB10, fitting the same 60 controls and evaluating the same 12 new
holdouts gives:

| Model | Control leave-size-out MAPE | New holdout MAPE | Holdout OOD |
| --- | ---: | ---: | ---: |
| Aggregate baseline | 14.9249% | 7.1441% | 0/12 |
| Program distribution | 5.8672% | Not evaluated | — |
| Dependency distribution, selected on controls | 5.7367% | 5.6037% | 0/12 |
| Combined distribution | 5.8672% | Not evaluated | — |

Nested control CV, including feature-set selection, is **5.9078%**. Each outer
fold selects its feature set using only the other three size groups. One outer
fold selects program distribution; three select dependency distribution.
The final selection uses all four control groups. Numerically tied CV scores
(rounded to six decimal places) prefer fewer features.

The new holdout error reduction is 1.5404 percentage points, or approximately
21.6% relative. It is not uniform:

| Holdout family, three sizes each | Aggregate MAPE | Selected MAPE |
| --- | ---: | ---: |
| LayerNorm | 6.6413% | 4.1921% |
| LogSoftmax | 4.9783% | 6.7267% |
| Residual SiLU | 8.4121% | 5.5059% |
| Independent mean/max joined in an expression | 8.5448% | 5.9899% |

No holdout result was used to refit or select another candidate. LogSoftmax's
regression remains in the reported metric.

## What changed

Experiment code now lives in `microbench/gpu/tests/primitive/kernels.py` and
`microbench/gpu/tests/compositional/kernels.py`; case membership is declared in
separate control/holdout JSON files under `microbench/gpu/configs/`. Timing is in
`microbench/gpu/harness/measure.py`. This is a directory-only reorganization of
the recorded experiment; the archived source snapshot retains its original paths.

`performance/gpu_distributions.py` derives features from existing CPU-interpreted
source events, without target compilation or instruction traces:

- Per-program demand for arithmetic warps, special-function warps, reduction
  tree depth, and tensor FLOPs; summarize each across programs.
- Component-wise longest **exposed source dependency path**, with unit arithmetic
  and special-function weights and `ceil(log2(reduction_width))` reduction weight.
  Layout operations propagate predecessor state at zero modeled cost.
- Retain mean, p50, p90, and max diagnostics; candidates use p90 features only.

The original seven aggregate features remain. Three alternative candidates add
four program features, three path features, or both. Calibration continues to
use nonnegative relative-squared-error fitting; selection reports MAPE. There is
no MLP and no claim that these structural weights are physical cycle latencies.

The selected model's nonzero new terms are reduction-path and SFU-path depth;
the arithmetic-path coefficient is zero. In these uniform-program experiments,
percentiles collapse to the same value. Thus this experiment supports retaining
local dependency/work depth in addition to global totals; it does **not** establish
the benefit of distribution tails on heterogeneous workloads.

`gpu.predict` recognizes the frozen `feature_set` and validates its feature
schema. Existing aggregate calibrations still use their original features and
prediction path. `GpuBackend` and `predict_latency` need no new public interface.
NKI's calibration and scheduling implementations are unchanged.

## Experiment protocol

The compositional suite has the original 36 controls plus 24 repeated sum/max
controls: four input sizes (4,096; 16,384; 65,536; 262,144), two reduction kinds,
and one/two/four repetitions. The 12 fresh holdouts use four source bodies and
three interleaved sizes (12,288; 49,152; 196,608). FP32 vector/reduction kernels
use block 512, four warps and two stages. Original dot controls retain their
original FP16 geometry.

Sequence: declare all cases → collect 60 controls → fit and freeze candidate
selection → collect 12 holdouts → evaluate only baseline and selected candidate.
The fit reader opens only declared control artifacts; a unit test enforces this.
Promotion requires both ordinary and nested CV MAPE at most 20%.

Timing remains CUDA-graph steady-cache per-kernel latency, not cold-cache or
host launch latency. Direct NVML monitoring allowed the pre-existing idle desktop
graphics processes, but rejected foreign compute activity and unstable batches.
Two control batches exceeded the 15% sample-span threshold and were retried;
all rejected attempts are retained. Accepted batches were marked uncontaminated
by this monitor, which does not prove exclusive GPU access. No other users'
processes or machine settings were changed. There is no independent-session
confidence interval; the modest holdout improvement needs a reserved-machine
replication. The environment also emitted a PyTorch advertised-capability warning
for GB10; all measured kernels passed the implemented reference-output checks.

## Artifacts and reproduction

Local artifacts (ignored by Git):
`downloads/gpu_concorde_20260911/run_v1/` contains the manifest, source observations,
measurements, and all attempts. `ablation_v1/` contains `frozen_ablation.json`,
deployable `frozen_model.json`, and per-case `evaluation.json`.
`results.tar.gz` archives both directories.
`measurement_source.tar.gz` preserves the source snapshot used for collection;
a subsequent import-only lint cleanup does not change model semantics but does
change the package source fingerprint for future collections.

Run fingerprint: `e98e9e457973940ff150`.
Frozen ablation digest: `08007615a3d41474c6f8`.

```bash
python -m triton_viz.tools.gpu_cost_model_pipeline collect --root NEW_RUN --suite compositional --role control
python -m triton_viz.tools.gpu_distribution_experiments fit --root NEW_RUN --output NEW_ABLATION
python -m triton_viz.tools.gpu_cost_model_pipeline collect --root NEW_RUN --suite compositional --role holdout
python -m triton_viz.tools.gpu_distribution_experiments evaluate --root NEW_RUN --output NEW_ABLATION
```

Only add `--allow-idle-graphics` to collection when intentionally accepting the
documented shared-desktop timing mode. Use fresh output directories per experiment;
do not reuse an older frozen model after a failed fit. For CPU prediction, load
`NEW_ABLATION/frozen_model.json` as the existing `GpuBackend` calibration.

Earlier exploratory replay of the previous 36-control/15-holdout pilot selected
program distribution: control CV 13.04% → 6.28%; old holdout MAPE 13.23% → 5.76%.
However, three of those old holdouts were OOD for the new reduction feature.
That all-case figure is not an in-domain result and is not mixed with this new
experiment. Expanded reduction controls were collected before the new holdouts.

Validation: 91 local focused performance/NKI tests passed, and 16 remote
distribution/interpreter tests passed. Ruff checks passed for the new modules
and tests. Tests include equal-total-work serial versus parallel paths, unequal
program-work percentiles, missing/cross-program dependency rejection, control-only
selection, public-backend prediction, and CPU reference checks for new kernels.

## Interpretation and next experiments

Concorde combines analytically derived component performance distributions with
learned fusion; its CPU simulator-based accuracy is not directly comparable to
our source-only GPU timing task. See the [paper](https://arxiv.org/abs/2503.23076).

Our next predeclared control axes should separate dependency chains from parallel
branches at equal work, vary tile size and warp count, and distinguish cache-resident
from streaming traffic. The traces used for these results omit some store-value
and dot accumulator edges; newly collected traces include them through the raw
operand hook. These results have not been rerun with the updated traces.
Register occupancy, spills, compiler contraction, memory hazards,
and physical layouts remain unmodeled. Marginal feature-range coverage does not
prove joint coverage. More complex fusion should only be considered after broader
controls and untouched application holdouts establish a repeatable advantage.
