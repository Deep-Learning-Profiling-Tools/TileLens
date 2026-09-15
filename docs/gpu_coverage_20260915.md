# GB10 coverage experiment, 2026-09-15

## Fixed protocol and provenance

- CPU paired controls: commit `f664066`; independent validation: `34bcf67`.
- 480 controls: 416 coverage declarations plus 64 paired single-program cases.
- 32 fresh validation cases, declared before any new GPU measurements.
- Only manifest-declared controls enter fitting. A runtime JSON-read guard logs
  all fit reads and refuses paths outside the control manifest/directory.
- Candidate selection uses grouped nested control CV and the existing 20% gate.
  No failing calibration is eligible for public prediction. No holdout fitting,
  target-specific constants, point deletion or target-driven candidate selection.
- Source observation precedes target compilation in collection/prediction.
  Control compilation checks are feasibility diagnostics, not model features.
- NVML-monitored shared desktop on GB10, with existing idle graphics processes
  explicitly allowed. This is not an exclusive reservation or cold-cache timing.

Remote artifacts are under `~/tmp/gpu_coverage_20260915/` on the user-provided
GB10. Python is the earlier experiment's `triton-viz-performance.IQZjVx/env`
environment (Torch 2.9.1+cu128, Triton 3.7.0). The existing PyTorch capability
warning for GB10 remains; individual numerical reference checks are enforced.

## Resource failure and versioned correction

The first `run/` stopped after 99 accepted controls: the large FP32/TF32 dot with
five iterations and three stages requested 131072 bytes of shared memory,
exceeding the device limit of 101376 bytes. That run and its attempts are retained.

All 32 large-dot declarations (128x128x64, across dtype, precision, repetition,
program count) now use one stage, uniformly chosen for resource feasibility.
No shape or case was dropped. All other control settings are unchanged.
The new `src_v2/` snapshot and `run_v2/` use a fresh source fingerprint; all 480
measurements are recollected rather than merging selected old timings.

Resource/numerical preflight passed for all 16 dot signatures at two programs
and all 64 paired controls. Large FP32 dots require 65536 shared bytes; large
FP16/BF16 dots require 32768. Compilation metadata is not passed to fitting.

## Results

Collection and gated fitting are in progress. No new 254-point MAPE is claimed
until complete, gated evaluation is available. The historical 18.5475% remains
a separate frozen-model development-set diagnostic.
