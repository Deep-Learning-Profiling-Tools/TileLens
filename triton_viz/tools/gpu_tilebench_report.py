"""Merge an original 254-point attempt and an explicitly labeled repair run."""

import argparse
import csv
from collections import Counter
from pathlib import Path

from triton_viz.tools.gpu_cost_model_pipeline import _read, _write
from triton_viz.tools.gpu_tilebench_evaluate import summarize
from triton_viz.performance.calibration import stable_digest


def merge(primary, supplement, output):
    first, second = (_read(p / "manifest.json") for p in (primary, supplement))
    for key in (
        "cases",
        "calibration_digest",
        "calibration_fingerprint",
        "tilebench_sources",
    ):
        if first[key] != second[key]:
            raise ValueError(f"Incompatible experiments: {key}")
    rows = []
    for case in first["cases"]:
        original = _read(primary / "cases" / (case["id"] + ".json"))
        row = dict(original)
        row.setdefault("implementation", "unmodified_tilebench")
        row["measurement_source"] = str(primary)
        row["prediction_source"] = str(primary)
        if "error_pct" not in row:
            repaired = _read(supplement / "cases" / (case["id"] + ".json"))
            if "error_pct" not in repaired:
                raise ValueError(f"Repair incomplete: {case['id']}")
            row = {
                **repaired,
                "original_failure": original.get("error"),
                "prediction_source": str(supplement),
                "measurement_source": str(supplement),
            }
            # First accepted hardware result wins, independent of its error.
            # BF16 mul2 needed only a CPU observation repair, not new timing.
            if "measured_us" in original:
                row["measured_us"] = original["measured_us"]
                row["measurement_source"] = str(primary)
            row["error_pct"] = 100 * (row["predicted_us"] / row["measured_us"] - 1)
        rows.append(row)
        _write(output / "cases" / (case["id"] + ".json"), row)
    _write(
        output / "manifest.json",
        dict(
            primary=str(primary),
            supplement=str(supplement),
            calibration_digest=first["calibration_digest"],
            cases=first["cases"],
            merge_policy="original predictions when available; repaired failures; earliest accepted hardware measurement",
        ),
    )
    report = summarize(output, first["cases"])
    report["ood_reason_counts"] = dict(
        Counter(reason for r in rows for reason in set(r["ood_reasons"]))
    )
    report["implementation_counts"] = dict(Counter(r["implementation"] for r in rows))
    for key in ("dtype", "rows"):
        report["by_" + key] = {
            str(value): dict(
                count=len(group := [r for r in rows if r["case"][key] == value]),
                mape_pct=sum(abs(r["error_pct"]) for r in group) / len(group),
            )
            for value in sorted({r["case"][key] for r in rows})
        }
    report[
        "interpretation"
    ] = "Diagnostic extrapolation, not an in-domain accuracy claim; nine explicitly adapted implementations"
    _write(output / "report.json", report)
    with (output / "cases.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "id",
                "op",
                "rows",
                "cols",
                "dtype",
                "implementation",
                "predicted_us",
                "measured_us",
                "error_pct",
                "ood_reasons",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row["case"],
                    **{k: row[k] for k in writer.fieldnames if k not in row["case"]},
                }
            )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-ablation", type=Path)
    args = parser.parse_args()
    report = merge(args.primary, args.supplement, args.output)
    if args.baseline_ablation:
        from triton_viz.performance.gpu import predict

        ablation = _read(args.baseline_ablation)
        digest = ablation.pop("digest")
        manifest = _read(args.primary / "manifest.json")
        if (
            digest != stable_digest(ablation)
            or ablation["fingerprint"] != manifest["calibration_fingerprint"]
        ):
            raise ValueError("Baseline ablation provenance mismatch")
        baseline_rows = []
        for case in manifest["cases"]:
            row = _read(args.output / "cases" / (case["id"] + ".json"))
            source = _read(
                Path(row["prediction_source"]) / "sources" / (case["id"] + ".json")
            )
            prediction = predict(
                source,
                ablation["models"]["aggregate"],
                fingerprint=ablation["fingerprint"],
                sm_count=manifest["identity"]["sm_count"],
                strict=False,
            )
            baseline_rows.append(
                dict(
                    case=case,
                    predicted_us=prediction["latency_us"],
                    measured_us=row["measured_us"],
                    ood_reasons=prediction["ood_reasons"],
                    error_pct=100 * (prediction["latency_us"] / row["measured_us"] - 1),
                )
            )
        report["aggregate_baseline_mape_pct"] = sum(
            abs(r["error_pct"]) for r in baseline_rows
        ) / len(baseline_rows)
        _write(
            args.output / "baseline.json",
            dict(frozen_ablation_digest=digest, cases=baseline_rows),
        )
        _write(args.output / "report.json", report)
    print(report)


if __name__ == "__main__":
    main()
