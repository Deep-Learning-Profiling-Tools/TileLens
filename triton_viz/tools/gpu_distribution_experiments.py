"""Control-only feature ablation; frozen evaluation never selects a candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from triton_viz.performance.calibration import fit_controls, stable_digest
from triton_viz.performance.gpu import expand, predict, source_configuration
from triton_viz.performance.gpu_distributions import (
    FEATURE_SETS,
    LEGACY_FEATURE_SETS,
    distribution_features,
)
from triton_viz.performance.gpu_dot_precision import dot_features
from triton_viz.tools.gpu_cost_model_pipeline import _read, _write


def enrich(source, sm_count):
    work = expand(source, sm_count=sm_count)
    if work["ood_reasons"]:
        raise ValueError(f"Unsupported source: {work['ood_reasons']}")
    features, distributions = distribution_features(source)
    return {**work["features"], **features}, source_configuration(work), distributions


def fit_candidates(rows, names, fingerprint):
    """Numerical rejection is candidate-local; invalid data must fail closed."""
    models, rejected = {}, {}
    for name in names:
        try:
            models[name] = fit_controls(
                rows, FEATURE_SETS[name], fingerprint=fingerprint
            )
        except ValueError as exc:
            if str(exc) != "Nonnegative calibration did not converge":
                raise
            rejected[name] = {"reason": "nonconvergence", "message": str(exc)}
    if not models:
        raise ValueError(f"All candidates failed numerical convergence: {rejected}")
    return models, rejected


def fit(root, output, *, dot_precision=False, memory_working_set=False, wave_dot=False):
    memory_working_set = memory_working_set or wave_dot
    dot_precision = dot_precision or memory_working_set
    manifest = _read(root / "manifest.json")
    rows = []
    configurations = []
    dot_configurations = []
    candidate_names = (
        ("dot_precision_aggregate", "dot_precision_combined")
        if dot_precision
        else LEGACY_FEATURE_SETS
    )
    if memory_working_set:
        candidate_names = (
            "dot_precision_combined",
            "dot_precision_memory",
            "dot_precision_memory_pressure",
        )
    if wave_dot:
        candidate_names += ("dot_precision_wave", "dot_precision_memory_wave")
    for case in manifest["splits"]["control"]:
        row = _read(root / "controls" / (case["id"] + ".json"))
        features, configuration, _ = enrich(
            row["source"], manifest["identity"]["sm_count"]
        )
        if dot_precision:
            extra, dot_configs, reasons = dot_features(row["source"])
            if reasons:
                raise ValueError(f"Unsupported control dot metadata: {reasons}")
            features.update(extra)
            for config in dot_configs:
                if config not in dot_configurations:
                    dot_configurations.append(config)
        if memory_working_set:
            from triton_viz.performance.gpu_memory import memory_features

            extra, reasons = memory_features(row["source"])
            if reasons:
                raise ValueError(f"Unsupported control memory metadata: {reasons}")
            features.update(extra)
        if wave_dot:
            from triton_viz.performance.gpu_dot_precision import wave_dot_features

            features.update(wave_dot_features(features))
        if wave_dot:
            # Hash complete source once, rather than serializing giant traces
            # again for every nested fit's provenance digest. Numeric inputs,
            # every measurement and the full source on disk remain unchanged.
            row = {
                **{k: v for k, v in row.items() if k != "source"},
                "source_digest": stable_digest(row["source"]),
            }
        rows.append({**row, "features": features})
        if configuration not in configurations:
            configurations.append(configuration)
    models, rejected = fit_candidates(rows, candidate_names, manifest["fingerprint"])
    for name, model in models.items():
        model["feature_set"] = name
        model["source_configurations"] = configurations
        if dot_precision:
            model["dot_configurations"] = dot_configurations

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
        inner, inner_rejected = fit_candidates(
            training, candidate_names, manifest["fingerprint"]
        )
        chosen = select(inner)
        scores = {name: model["cv"]["mape_pct"] for name, model in inner.items()}
        ranked = sorted(scores.values())
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
                "rejected_candidates": inner_rejected,
                "selected": chosen,
                "candidate_cv_mape_pct": scores,
                "selection_margin_pct": ranked[1] - ranked[0]
                if len(ranked) > 1
                else None,
                "errors_pct": errors,
                "mape_pct": float(np.mean(np.abs(errors))),
            }
        )
    result = {
        "schema": "triton-viz.gpu-distribution-experiment.v1",
        "fingerprint": manifest["fingerprint"],
        "models": models,
        "rejected_candidates": rejected,
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
    if (
        not frozen["models"][selected]["cv"]["passed"]
        or frozen["nested_mape_pct"] > 20.0
    ):
        raise ValueError("Selected calibration failed the control/nested CV gate")
    reports = {}
    baseline = (
        "dot_precision_combined"
        if "dot_precision_memory" in frozen["models"]
        else (
            "dot_precision_aggregate"
            if selected.startswith("dot_precision_")
            else "aggregate"
        )
    )
    for name in dict.fromkeys((baseline, selected)):
        if name not in frozen["models"]:
            reports[name] = {
                "status": "rejected_control_numerics",
                "reason": frozen.get("rejected_candidates", {}).get(name),
            }
            continue
        model = frozen["models"][name]
        if not model["cv"]["passed"]:
            reports[name] = {"status": "rejected_control_cv", "cv": model["cv"]}
            continue
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
    parser.add_argument("--wave-dot", action="store_true")
    parser.add_argument(
        "--memory-working-set",
        action="store_true",
        help="Compare preregistered precision/memory candidates on controls only",
    )
    parser.add_argument(
        "--dot-precision",
        action="store_true",
        help="Fit only precision-separated candidates; requires newly observed dot metadata",
    )
    args = parser.parse_args(argv)
    if args.stage == "fit":
        fit(
            args.root,
            args.output,
            dot_precision=args.dot_precision,
            memory_working_set=args.memory_working_set,
            wave_dot=args.wave_dot,
        )
    else:
        evaluate(args.root, args.output)


if __name__ == "__main__":
    main()
