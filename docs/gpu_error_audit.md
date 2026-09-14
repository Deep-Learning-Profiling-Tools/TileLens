# GPU 254-point development-set error audit

Date: 2026-09-14. This audit reads the manifest-declared cases in
`downloads/gpu_tilebench254_20260911/merged/`, recomputes errors from the frozen
predictions and first accepted hardware measurements, and performs no fitting,
model selection, or retiming. The 254 cases are now a development set.

## Findings

Reproduced MAPE: **18.547505%**, with **254/254 OOD**, **226/254 underpredictions**,
and mean signed percentage error **-17.100626%**. The metric has not improved in
this audit. A global correction factor would be target-based fitting and would
not resolve unsupported operations or precision semantics.

Error contribution is `sum(abs(error_pct)) / 254`. These contributions are
additive across disjoint operator groups; they are not causal estimates or
promised improvements.

| Family | Cases | MAPE | Contribution to total MAPE (pp) |
| --- | ---: | ---: | ---: |
| KL divergence | 30 | 30.55% | 3.6079 |
| LayerNorm | 30 | 28.20% | 3.3308 |
| RMSNorm | 30 | 21.80% | 2.5748 |
| Softmax | 30 | 21.35% | 2.5214 |
| Interleave | 30 | 16.33% | 1.9289 |
| Attention adapter | 4 | 88.74% | 1.3974 |
| Sigmoid | 30 | 7.61% | 0.8988 |
| Matmul | 10 | 21.94% | 0.8636 |
| Mul2 | 30 | 6.94% | 0.8193 |
| ReLU | 30 | 5.12% | 0.6046 |

The first four families contribute **12.0350 pp**, about **64.89%** of total
absolute percentage error. Even eliminating attention's error entirely, with
everything else unchanged, would reduce total MAPE by only 1.3974 pp to 17.1501%.
This arithmetic ceiling makes reductions and local latency the higher-priority
experiment for the all-case metric.

| Family | 1 row MAPE | 16 rows MAPE | 128 rows MAPE |
| --- | ---: | ---: | ---: |
| KL divergence | 36.07% | 33.43% | 22.15% |
| LayerNorm | 36.80% | 32.32% | 15.48% |
| RMSNorm | 28.76% | 26.73% | 9.91% |
| Softmax | 28.02% | 26.17% | 9.85% |

All cases in the 1-row and 16-row cells above underpredict. This is consistent
with insufficient modeling of local serial latency at low parallelism, but
kernel structure, launch geometry and omitted work confound that interpretation.
Logical rows are not necessarily launched program counts.

## OOD coverage

Each reason below counts affected cases once. Categories overlap.

| Reason | Cases |
| --- | ---: |
| Uncovered launch/tile/dtype configuration | 254 |
| Global-sector range | 103 |
| Tensor-FLOP range | 11 |
| Shuffle-step range | 2 |
| Unpriced ternary operation | 70 |
| Unpriced integer division | 10 |
| Unpriced greater comparison | 30 |
| Unpriced greater-equal comparison | 30 |
| Unmodeled join layout conversion | 30 |

Retain these gates. Merely expanding feature bounds or whitelisting operations
would not establish coverage or supply missing prices. ReLU's low error despite
unpriced comparison/selection also shows that an OOD label is not a causal error
attribution: source operations may fuse or their costs may already be absorbed
by other fitted terms.

## Next experiment

1. Reuse the existing **416 control declarations** in
   `microbench/gpu/configs/coverage_control.json`. They span 2/8/32/192 programs,
   loop depths, dtype/launch geometry, and IEEE/TF32 dots. Add independent
   low-parallelism and equal-total-work serial/parallel controls before collecting
   new timings. Include one-program controls, currently absent. Define fresh
   validation kernels and sizes before looking at their measurements.
2. Reobserve controls with the repaired store-value/dot-accumulator dependency
   hook. Refit only controls and freeze using grouped, nested CV. The archived
   model and traces predate those edge repairs; replaying targets alone is not
   enough to validate the changed feature extraction.
3. Test local path/work terms on reduction controls first. Inspect whether
   semantic dot operand dtype and input precision are retained: current pricing
   pools tensor FLOPs, and the source event does not record `input_precision`.
   IEEE, TF32 and BF16 therefore need explicit observation metadata before a
   precision-specific model can be calibrated. Do not infer hardware lowering
   solely from the output dtype.
4. Add isolated comparison/selection and layout controls to determine whether
   separate prices improve control CV. Use compiler evidence from controls for
   fusion/layout hypotheses; keep the source-only prediction interface intact.
5. Evaluate the frozen candidate on fresh validation first, then on these 254
   development cases. Report MAPE, signed bias, all-case OOD counts and missing
   operation prices together. Recollect on GB10 under the same timing protocol;
   no GPU measurements were made in this audit (local `nvidia-smi` unavailable).

## Reproduction

```bash
python -m triton_viz.tools.gpu_error_audit \
  --root downloads/gpu_tilebench254_20260911/merged \
  --output downloads/gpu_tilebench254_20260911/error_audit_20260914.json
```

Use a fresh output path on rerun. The JSON includes dtype, row, operator/row,
operator/dtype and overlapping OOD groups. The tool ignores undeclared artifacts,
rejects invalid latencies and case identities, and never reads control data.
