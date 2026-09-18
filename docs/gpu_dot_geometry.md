# Preregistered dot geometry / reuse / pipeline diagnostic

## Motivation and scope

The preceding precision experiment already includes 128x128x64 dots with
2/8/32/192 programs, four precision classes and 17/65 loop iterations.
Consequently, absence of large dots is not an established explanation for the
remaining matmul error. Precision-only features cannot distinguish tile shape,
cross-program operand reuse or pipeline stage depth at equal source FLOPs.

This follow-up diagnoses those factors using **only new controls**. It does not
load the 254-point measurements, modify holdouts, refit coefficients or select
models using target errors. The existing 512-control frozen model is unchanged.

## Frozen matrix before GPU timing

`microbench/gpu/configs/geometry_control.json` declares all 144 controls:

- FP32 IEEE, FP32 TF32, BF16 and FP16, FP32 accumulation throughout;
- tiles 64x64x32, 128x128x32 and 128x128x64;
- 8 and 64 programs, 17 dependent dot iterations, eight warps;
- disjoint inputs, shared A only, shared A and B across programs;
- one and two compiler pipeline stages.

All programs receive identical numerical inputs even when storage is disjoint.
The six reuse/stage variants of each geometry/precision/program configuration
form one CV group. Sharing changes address overlap and allocation footprint,
not FLOPs, number of source loads, output work or accumulator dependencies.
This is not a complete tiled-matmul scheduling simulation: no inference that
reuse alone explains target errors is warranted.

The main analysis is matched latency ratios across reuse and stage variants,
reported separately by precision and tile. No point filtering by latency is
allowed. Resource failures remain failures in the declared matrix, not removed
cases. Compiler resource metadata, if inspected to explain failures, cannot
become pre-compile model inputs.

## Protocol

Use the unchanged monitored steady-cache CUDA-graph harness on GB10. CPU source
observation and reference checks precede GPU compilation. Accepted batches must
pass existing contamination checks; all rejected attempts remain archived.
Idle desktop graphics are explicitly permitted, not claimed exclusive access.
Expected window: 20–60 minutes, depending on observation and compilation costs.

```bash
python -m triton_viz.tools.gpu_cost_model_pipeline collect \
  --root NEW_GEOMETRY_RUN --suite geometry --role control --allow-idle-graphics
```

This separate run contains 144 **additional** controls and no holdouts. Do not
merge it into the prior 512 rows by spoofing source fingerprints. Any subsequent
fit requires an explicit audited multi-run provenance policy or recollection,
control-only feature selection and the unchanged ordinary/nested CV gates.

## Completed control-only result

The GB10 run completed all **144/144** declared controls: 144 measurement
attempts, zero rejected batches, no resource failures and no deleted points.
The first-to-last measurement telemetry spans 05:12:44–05:18:44 UTC on
September 18, about six minutes, shorter than the conservative window estimate.
All accepted rows match manifest fingerprint `d475f301a40ef1f2487a`, their
declared cases, roles and CV groups. CPU numerical checks precede compilation;
GPU numerical checks also passed. The remote environment's 13 precollection
CPU tests passed. The final local focused suite passes 77 tests, including
control-only audit read boundaries and rejection of an incomplete matrix.
Ruff and diff checks pass.

The complete 24-group audit reports identical **aggregate feature vectors**
within every six-way reuse/stage group. Precision and dot FLOPs also remain
fixed within each group. Nevertheless, latency differs substantially in the
large working-set cases. For 64 programs, 128x128x64, 17 iterations, stage 1:

| Input class | Disjoint inputs (us) | Shared A (us) | Shared A/B (us) |
| --- | ---: | ---: | ---: |
| IEEE FP32 | 406.86 | 289.34 | 229.14 |
| TF32 | 292.23 | 204.61 | 106.46 |
| BF16 | 171.82 | 55.54 | 47.02 |
| FP16 | 167.12 | 48.65 | 46.66 |

For BF16/FP16, the allocated input working set falls from 34 MiB (disjoint)
to 17.265625 MiB (shared A) and 0.53125 MiB (shared A/B). For FP32 these
sizes double. Source load requests, accumulation chains and numerical values
are unchanged. This demonstrates that a single price for dot FLOPs plus
per-request sectors cannot distinguish these controls. Cache/working-set
effects are a plausible explanation, not a measured hardware-cache attribution.

The effect is **not universal**: across all 24 groups, median shared-A/disjoint
latency is 0.9804 at stage 1 and 0.9895 at stage 2; shared-A/B ratios are 0.9709
and 0.9780. For stage 2/stage 1, medians are 1.0194 (disjoint), 0.9841 (shared A)
and 1.0017 (shared A/B). Therefore neither a fixed reuse discount nor a fixed
pipeline speedup is justified. The large-dot rows above are a predeclared
subgroup; the artifact includes all groups, not only the favorable examples.

The tile sweep changes work and memory footprint as well as tile geometry; it
does not by itself estimate a pure equal-FLOP geometry effect. Measurements are
one monitored session in declaration order, not randomized independent-session
replicates. Small ratios near one should not be treated as significant wins.

### Consequences for subsequent modeling

Current `expand()` sums sectors per access, while `triton_observe` retains only
each access's sector count, not cross-program address overlap. Pipeline stages
and tile shapes are coverage metadata, not independent fitted prices. The new
controls establish a representational gap; simply adding their timing rows to
the same feature matrix cannot distinguish matched reuse variants.

The next control-only modeling step should expose source-derived unique load
footprint / repeated-sector demand and retain tile and precision context,
before fitting any reuse-dependent terms. Existing control traces would need
compatible reobservation/provenance because their cross-program addresses have
already been discarded. Further matched equal-work tile controls and repeated
sessions can separate geometry from memory footprint and session effects.
All candidate choices must use controls and pass unchanged CV gates. No claim
is made that these controls have repaired the historical matmul predictions.

**No fit, target read, holdout edit or 254-point evaluation occurred in this
experiment.** The previous frozen model and 15.6163% historical replay remain
unchanged. This is a diagnosis, not a new MAPE improvement.

Artifacts are retained in `downloads/gpu_geometry_20260917/`: `audit.json`,
`run/` (controls and all measurement attempts), `collect.log`,
`controls_complete.tar.gz`, and exact base/overlay source archives. The run ID
was declared on September 17; collection completed on September 18 UTC.
Remote artifacts remain under `~/tmp/gpu_geometry_20260917/`.

Reproduce the audit without opening any target files:

```bash
python -m triton_viz.tools.gpu_dot_geometry_audit \
  --root downloads/gpu_geometry_20260917/run \
  --output downloads/gpu_geometry_20260917/audit.json
```
