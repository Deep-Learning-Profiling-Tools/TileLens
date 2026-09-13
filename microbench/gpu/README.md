# Triton/GPU microbenchmarks

GPU experiments mirror the responsibilities of `microbench/inf2_nki/`.
The prediction library remains in `triton_viz/performance/`; this directory is
research/benchmark infrastructure, run from a repository checkout with GPU
dependencies installed. It is not required to load a frozen prediction model.

```text
microbench/
├── inf2_nki/                    existing NKI experiments, unchanged
└── gpu/
    ├── configs/
    │   ├── pilot_control.json          36 controls
    │   ├── pilot_holdout.json          15 holdouts
    │   ├── compositional_control.json  60 controls
    │   ├── compositional_holdout.json  12 holdouts
    │   └── tilebench254_holdout.json    holdout-only reference to NKI case list
    ├── common/cases.py          role-checked declarations, no CUDA imports
    ├── tests/
    │   ├── primitive/kernels.py
    │   └── compositional/kernels.py
    ├── harness/
    │   ├── run_microbench.py    audited collection entry point
    │   └── measure.py          CUDA graphs, NVML monitoring, batch rejection
    └── holdouts/attention.py    explicit unequal-V attention adapter
```

As in the NKI tree, `tests/` contains benchmark kernels, not pytest test cases.
Regression tests remain in `tests/unit/test_gpu_*.py` and performance tests.
Primitive kernel bodies retain the original fixed modes; membership in the
control versus holdout split is declared in separate configuration files.

| Responsibility | NKI | GPU |
| --- | --- | --- |
| Experiment declaration | `inf2_nki/configs/` | `gpu/configs/` |
| Control kernels | `inf2_nki/tests/*/kernels.py` | `gpu/tests/*/kernels.py` |
| Collection harness | `inf2_nki/harness/` | `gpu/harness/` |
| Shared experiment helpers | `inf2_nki/common/` | `gpu/common/` |
| Pipeline/fitting/target evaluation | `triton_viz/tools/nki_*.py` | `triton_viz/tools/gpu_*.py` |
| Prediction API | `NkiBackend` | `GpuBackend` |

NKI additionally has `profile_parser/` for compiler/hardware profile formats.
GPU currently stores audited event timings directly; there is no GPU equivalent
parser to move or duplicate. Alignment means matching responsibilities, not
creating empty directories or importing NKI hardware semantics into GPU code.

## Control/holdout boundary

`load_cases(suite, "control")` opens only that suite's control JSON. It cannot
select `tilebench254` as a control suite. The 254 declaration references the
canonical NKI `formal_holdouts.json`; deduplication still produces exactly 254
logical cases. No NKI cases or configurations are changed by this reorganization.

Fit stages continue to read only `controls/` under the experiment output root.
Target evaluation never fits. The 254 points remain holdouts; seeing their errors
does not authorize training on their measured latencies. Nine explicit GPU
implementation adapters remain labeled as described in the
[254-point report](../../docs/gpu_tilebench254_results.md).

## Commands

Run from the repository root:

```bash
# New directory-local collection entry point (equivalent to pipeline collect).
python -m microbench.gpu.harness.run_microbench \
  --root NEW_RUN --suite compositional --role control
python -m triton_viz.tools.gpu_distribution_experiments fit \
  --root NEW_RUN --output NEW_ABLATION
python -m microbench.gpu.harness.run_microbench \
  --root NEW_RUN --suite compositional --role holdout
python -m triton_viz.tools.gpu_distribution_experiments evaluate \
  --root NEW_RUN --output NEW_ABLATION
```

Existing `python -m triton_viz.tools.gpu_cost_model_pipeline collect/fit/evaluate`
commands remain available, just like the NKI pipeline. Use `--dry-run` for a
CPU-only CLI check. Shared-desktop allowance must be explicitly selected for
microbench collection; see [measurement protocol](../../docs/performance_backends.md).
The separate Tilebench evaluator retains its documented shared-desktop policy.

The new GPU source digest covers both `triton_viz/**/*.py` and relocated GPU
Python/JSON experiment files. Renaming these files changes the experiment
fingerprint: use a fresh collection root. Historical raw measurements, frozen
calibrations, and source archives are not rewritten. CPU-only fitting/evaluation
of archived data remains possible without recapturing hardware timings.

## Migration

| Previous internal location | Current location |
| --- | --- |
| `performance/gpu_controls.py` | `microbench/gpu/tests/primitive/kernels.py` |
| `performance/gpu_compositional_controls.py` | `microbench/gpu/tests/compositional/kernels.py` |
| `performance/gpu_measure.py` | `microbench/gpu/harness/measure.py` |
| `performance/gpu_tilebench_attention.py` | `microbench/gpu/holdouts/attention.py` |

Internal imports and current docs use the new locations. Public prediction
interfaces and frozen-model numerical behavior are unchanged. Archived source
snapshots intentionally retain their historical paths.
