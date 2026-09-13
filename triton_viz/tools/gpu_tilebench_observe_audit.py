"""CPU-only numerical observation audit, retaining all frozen hardware results."""

import argparse
from pathlib import Path

from triton_viz.performance.calibration import stable_digest
from triton_viz.tools.gpu_cost_model_pipeline import _read, _write
from triton_viz.tools.gpu_tilebench_evaluate import prepare, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("merged", "tilebench-dir", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    import torch
    from triton_viz.performance.gpu import predict
    from triton_viz.performance.triton_observe import observe

    model = _read(args.model)
    digest = model.pop("digest")
    assert stable_digest(model) == digest
    manifest = _read(args.merged / "manifest.json")
    assert manifest["calibration_digest"] == digest
    _write(
        args.output / "manifest.json",
        {**manifest, "audit": "CPU numeric replay, no hardware retiming or fit"},
    )
    diagnostics = []
    for case in manifest["cases"]:
        row = _read(args.merged / "cases" / (case["id"] + ".json"))
        (kernel, grid, inputs, kwargs), output, reference = prepare(
            case,
            args.tilebench_dir,
            adapters=row["implementation"] == "explicit_semantic_adapter",
        )
        source = observe(kernel, grid, *inputs, **kwargs)
        torch.testing.assert_close(
            output,
            reference,
            rtol=0.03 if case["dtype"] == "bfloat16" else 0.01,
            atol=0.02 if case["dtype"] == "bfloat16" else 0.002,
        )
        prediction = predict(
            source, model, fingerprint=model["fingerprint"], sm_count=48, strict=False
        )
        delta = prediction["latency_us"] - row["predicted_us"]
        diagnostics.append(dict(case=case, prediction_delta_us=delta))
        _write(args.output / "sources" / (case["id"] + ".json"), source)
        row.update(
            predicted_us=prediction["latency_us"],
            ood_reasons=prediction["ood_reasons"],
            prediction_source=str(args.output),
            cpu_numerical_validation="passed",
        )
        row["error_pct"] = 100 * (row["predicted_us"] / row["measured_us"] - 1)
        _write(args.output / "cases" / (case["id"] + ".json"), row)
        print(case["id"], "CPU pass", "prediction_delta_us", delta, flush=True)
    _write(args.output / "audit.json", diagnostics)
    print(summarize(args.output, manifest["cases"]), flush=True)


if __name__ == "__main__":
    main()
