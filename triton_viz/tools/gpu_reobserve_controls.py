"""Reobserve complete control runs, preserving measured-session provenance."""

from __future__ import annotations

import argparse
from pathlib import Path

from microbench.gpu.common.cases import load_cases
from microbench.gpu.tests.coverage.kernels import prepare, check_output
from triton_viz.performance.calibration import stable_digest
from triton_viz.performance.gpu import expand, source_configuration
from triton_viz.performance.triton_observe import observe
from triton_viz.tools.gpu_cost_model_pipeline import _read, _write, _source_digest


def reobserve(roots, output):
    import torch
    import triton
    import triton_viz
    from unittest.mock import patch

    parents, declarations = [], []
    for root in roots:
        manifest = _read(root / "manifest.json")
        suite = manifest["identity"]["suite"]
        if suite not in {"precision", "geometry", "structure", "stability"}:
            raise ValueError(
                "Only declared precision/geometry/structure/stability controls are supported"
            )
        cases = load_cases(suite, "control")
        if cases != manifest["splits"]["control"]:
            raise ValueError("Control declaration mismatch")
        if stable_digest(manifest["identity"]) != manifest["fingerprint"]:
            raise ValueError("Parent identity digest mismatch")
        parents.append(
            {
                "root": str(root.resolve()),
                "identity": manifest["identity"],
                "fingerprint": manifest["fingerprint"],
            }
        )
        declarations.extend((root, manifest, c) for c in cases)
    if len({c["id"] for _, _, c in declarations}) != len(declarations):
        raise ValueError("Duplicate controls across parent runs")
    for key in ("uuid", "sm_count", "driver", "packages", "metric", "capability"):
        if any(p["identity"][key] != parents[0]["identity"][key] for p in parents):
            raise ValueError(f"Incompatible measurement sessions: {key}")
    identity = {
        "protocol": "control-only CPU reobservation; unchanged source events and original timings",
        "sm_count": parents[0]["identity"]["sm_count"],
        "parents": parents,
        "observation_source_digest": _source_digest(),
        "cv_policy": "all dot controls with equal program count share a fold; paired controls unchanged",
    }
    fingerprint = stable_digest(identity)
    manifest = {
        "identity": identity,
        "fingerprint": fingerprint,
        "splits": {"control": [c for _, _, c in declarations], "holdout": []},
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and _read(manifest_path) != manifest:
        raise ValueError("Observation experiment changed; use a new output root")
    _write(manifest_path, manifest)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Control reobservation must not compile or initialize CUDA"
        )

    with patch.object(triton.compiler, "compile", forbidden), patch.object(
        torch.cuda, "_lazy_init", forbidden
    ):
        for root, parent, case in declarations:
            old_path = root / "controls" / (case["id"] + ".json")
            old = _read(old_path)
            if (
                old["role"] != "control"
                or old["case"] != case
                or old["fingerprint"] != parent["fingerprint"]
                or old["contaminated"]
            ):
                raise ValueError(f"Invalid measured control: {case['id']}")
            provenance = {
                "measurement_path": str(old_path.resolve()),
                "measurement_digest": stable_digest(old),
                "measurement_fingerprint": old["fingerprint"],
            }
            path = output / "controls" / (case["id"] + ".json")
            if path.exists():
                saved = _read(path)
                if (
                    saved["fingerprint"] != fingerprint
                    or saved["provenance"] != provenance
                ):
                    raise ValueError("Resume provenance mismatch")
                continue
            try:
                kernel, grid, args, out = prepare(case, "cpu")
                source = observe(
                    kernel,
                    grid,
                    *args,
                    num_warps=case.get("num_warps", 4),
                    num_stages=case.get("num_stages", 2),
                )
                check_output(case, out)
                legacy = {k: v for k, v in source.items() if k != "memory_working_set"}
                comparable = source if "memory_working_set" in old["source"] else legacy
                if comparable != old["source"]:
                    raise ValueError(f"Legacy source event mismatch: {case['id']}")
                work = expand(source, sm_count=identity["sm_count"])
                if work["features"] != old["features"] or work["ood_reasons"]:
                    raise ValueError("Legacy control work mismatch")
                group = (
                    str(case["programs"])
                    if case["kind"] in {"coverage_dot", "geometry_dot"}
                    else old["cv_group"]
                )
                _write(
                    path,
                    {
                        **old,
                        "source": source,
                        "fingerprint": fingerprint,
                        "cv_group": group,
                        "provenance": provenance,
                        "source_configuration": source_configuration(work),
                        "cpu_numerical_validation": "passed",
                        "legacy_source_validation": "identical",
                    },
                )
                print(
                    f"control {case['id']}: unchanged measurement, identical legacy source",
                    flush=True,
                )
            finally:
                triton_viz.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reobserve(args.control_run, args.output)


if __name__ == "__main__":
    main()
