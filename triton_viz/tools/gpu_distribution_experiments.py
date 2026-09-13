"""Control-only feature ablation; frozen evaluation never selects a candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from triton_viz.performance.calibration import fit_controls, stable_digest
from triton_viz.performance.gpu import expand, predict, source_configuration
from triton_viz.performance.gpu_distributions import FEATURE_SETS, distribution_features
from triton_viz.tools.gpu_cost_model_pipeline import _read, _write


def enrich(source, sm_count):
    work = expand(source, sm_count=sm_count)
    if work["ood_reasons"]:
        raise ValueError(f"Unsupported source: {work['ood_reasons']}")
    features, distributions = distribution_features(source)
    return {**work["features"], **features}, source_configuration(work), distributions


def fit(root, output):
    manifest = _read(root / "manifest.json")
    rows = []
    configurations = []
    for case in manifest["splits"]["control"]:
        row = _read(root / "controls" / (case["id"] + ".json"))
        features, configuration, _ = enrich(
            row["source"], manifest["identity"]["sm_count"]
        )
        rows.append({**row, "features": features})
        if configuration not in configurations:
            configurations.append(configuration)
    models = {
        name: fit_controls(rows, names, fingerprint=manifest["fingerprint"])
        for name, names in FEATURE_SETS.items()
    }
    for name, model in models.items():
        model["feature_set"] = name
        model["source_configurations"] = configurations

    def select(candidates):
        # Numerically indistinguishable CV scores prefer the smaller model.
        return min(
            candidates,
            key=lambda name: (
                round(candidates[name]["cv"]["mape_pct"], 6),
                len(FEATURE_SETS[name]),
                name,
            ),
        )

    selected = select(models)
    # Nested CV quantifies feature-selection optimism, using control data only.
    groups = sorted({r["cv_group"] for r in rows})
    if len(groups) < 4:
        raise ValueError("Nested model selection requires at least four control groups")
    nested = []
    for group in groups:
        training = [r for r in rows if r["cv_group"] != group]
        testing = [r for r in rows if r["cv_group"] == group]
        inner = {
            name: fit_controls(training, names, fingerprint=manifest["fingerprint"])
            for name, names in FEATURE_SETS.items()
        }
        chosen = select(inner)
        coefficients = inner[chosen]["coefficients_us"]
        errors = [
            100
            * (
                sum(r["features"][k] * v for k, v in coefficients.items())
                / r["latency_us"]
                - 1
            )
            for r in testing
        ]
        nested.append(
            {
                "group": group,
                "selected": chosen,
                "errors_pct": errors,
                "mape_pct": float(np.mean(np.abs(errors))),
            }
        )
    result = {
        "schema": "triton-viz.gpu-distribution-experiment.v1",
        "fingerprint": manifest["fingerprint"],
        "models": models,
        "selected": selected,
        "nested_cv": nested,
        "nested_mape_pct": float(
            np.mean([abs(e) for f in nested for e in f["errors_pct"]])
        ),
        "source_configurations": configurations,
        "selection_policy": "lowest control-only leave-size-group-out MAPE; nested CV reported",
    }
    result["digest"] = stable_digest(result)
    _write(output / "frozen_ablation.json", result)
    selected_model = {
        **models[selected],
        "selection_nested_mape_pct": result["nested_mape_pct"],
    }
    if not selected_model["cv"]["passed"] or result["nested_mape_pct"] > 20.0:
        raise ValueError(
            "Selected candidate failed the control/nested CV gate; not promoted"
        )
    selected_model["digest"] = stable_digest(selected_model)
    _write(output / "frozen_model.json", selected_model)
    print(
        json.dumps(
            {
                "selected": selected,
                "cv": {n: m["cv"] for n, m in models.items()},
                "nested_mape_pct": result["nested_mape_pct"],
            },
            indent=2,
        ),
        flush=True,
    )


def evaluate(root, output):
    manifest = _read(root / "manifest.json")
    frozen = _read(output / "frozen_ablation.json")
    digest = frozen.pop("digest")
    if (
        digest != stable_digest(frozen)
        or frozen["fingerprint"] != manifest["fingerprint"]
    ):
        raise ValueError("Frozen experiment identity mismatch")
    selected = frozen["selected"]
    reports = {}
    for name in ("aggregate", selected):
        model = frozen["models"][name]
        cases = []
        for case in manifest["splits"]["holdout"]:
            row = _read(root / "holdouts" / (case["id"] + ".json"))
            if (
                row["role"] != "holdout"
                or row["contaminated"]
                or row["fingerprint"] != manifest["fingerprint"]
            ):
                raise ValueError("Invalid holdout provenance")
            prediction = predict(
                row["source"],
                model,
                fingerprint=manifest["fingerprint"],
                sm_count=manifest["identity"]["sm_count"],
                strict=False,
            )
            predicted = prediction["latency_us"]
            cases.append(
                {
                    "case": case,
                    "predicted_us": predicted,
                    "measured_us": row["latency_us"],
                    "error_pct": 100 * (predicted / row["latency_us"] - 1),
                    "ood_reasons": prediction["ood_reasons"],
                    "contributions_us": prediction["contributions_us"],
                    "distributions": prediction["work"].get("distributions", {}),
                }
            )
        reports[name] = {
            "mape_pct": float(np.mean([abs(r["error_pct"]) for r in cases])),
            "ood_count": sum(bool(r["ood_reasons"]) for r in cases),
            "cases": cases,
        }
    _write(
        output / "evaluation.json",
        {
            "selected_before_evaluation": selected,
            "frozen_digest": digest,
            "models": reports,
        },
    )
    print(
        json.dumps(
            {
                n: {k: v for k, v in r.items() if k != "cases"}
                for n, r in reports.items()
            },
            indent=2,
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "evaluate"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    {"fit": fit, "evaluate": evaluate}[args.stage](args.root, args.output)


if __name__ == "__main__":
    main()
