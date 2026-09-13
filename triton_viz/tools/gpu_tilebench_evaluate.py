"""Evaluate the deduplicated NKI 254 shapes on unmodified Tilebench Triton kernels.

No fitting. Missing implementations, numerical failures and OOD remain visible.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

from triton_viz.performance.calibration import stable_digest
from triton_viz.tools.gpu_cost_model_pipeline import _read, _write, _identity


def cases(path):
    unique = {}
    for split in _read(path)["splits"].values():
        rows = split["rows"]
        for r in rows if isinstance(rows, list) else [rows]:
            for op, dims in split["operators"].items():
                for c in dims:
                    key = f"{op}_{r}_{c}_{split['dtype']}"
                    unique[key] = dict(
                        id=key, op=op, rows=r, cols=c, dtype=split["dtype"]
                    )
    return list(unique.values())


class Capture:
    def __init__(self, kernel, launches):
        self.kernel, self.launches = kernel, launches

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            # Triton default stage count is 3; make it explicit for observation.
            kwargs.setdefault("num_stages", 3)
            kwargs.setdefault("num_warps", 4)
            self.launches.append((self.kernel, grid, args, kwargs))

        return launch


def prepare(case, directory, *, adapters=False):
    import torch
    import triton

    op, r, c = case["op"], case["rows"], case["cols"]
    if op == "tiled_attention":
        if adapters:
            from microbench.gpu.holdouts.attention import attention

            generator = torch.Generator().manual_seed(0)
            q = torch.randn((128, 128), generator=generator)
            k = torch.randn((128, 128), generator=generator)
            v = torch.randn((128, c), generator=generator)
            output = torch.empty((128, c))
            reference = (q @ k.T / (128**0.5)).softmax(-1) @ v
            return (
                (
                    attention,
                    (4, c // 64),
                    (q, k, v, output),
                    dict(DV=c, num_warps=4, num_stages=2),
                ),
                output,
                reference,
            )
        raise ValueError(
            "No matching Tilebench tiled_attention implementation: flash_attention fixes V/output dimension to Q dimension"
        )
    path = directory / op / "impl_triton.py"
    spec = importlib.util.spec_from_file_location("tilebench_eval_" + op, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if adapters and op == "matmul_fp32_fp16_fp8" and case["dtype"] == "bfloat16":
        # Explicit FP16-geometry BF16 adapter, not a Tilebench default or tuning.
        module._DEFAULT_CONFIGS[torch.bfloat16] = dict(
            module._DEFAULT_CONFIGS[torch.float16]
        )
    launches = []
    for name, value in list(vars(module).items()):
        if isinstance(value, triton.runtime.JITFunction):
            setattr(module, name, Capture(value, launches))
    generator = torch.Generator().manual_seed(0)
    dtype = getattr(torch, case["dtype"])

    def rand(shape):
        return torch.randn(shape, generator=generator).to(dtype)

    x = rand((r, c))
    xf = x.float()
    if op == "matmul_fp32_fp16_fp8":
        y = rand((c, r))
        args, ref = (x, y), xf @ y.float()
    elif op == "interleave":
        y = rand((r, c))
        args, ref = (x, y, r * c), torch.stack((x, y), dim=-1).flatten()
    elif op == "kl_divergence":
        y = rand((r, c)).abs()
        args = (x, y)
        ref = (y.float() * (y.float().clamp_min(1e-30).log() - xf)).sum(-1)
    elif op == "layernorm":
        w, b = rand((c,)), rand((c,))
        args = (x, w, b)
        ref = torch.nn.functional.layer_norm(xf, (c,), w.float(), b.float(), 1e-5)
    elif op == "rmsnorm":
        w = rand((c,))
        args = (x, w)
        ref = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6) * w.float()
    else:
        args = (x, r * c) if op == "sigmoid" else (x,)
        ref = {
            "relu": lambda: xf.relu(),
            "mul2": lambda: xf * 2,
            "sigmoid": lambda: xf.sigmoid(),
            "softmax": lambda: xf.softmax(-1),
        }[op]()
    output = module.run(*args, autotune=False)
    if len(launches) != 1:
        raise ValueError(f"Expected one kernel launch, got {len(launches)}")
    return launches[0], output, ref.to(output.dtype)


def summarize(root, declared):
    rows = [
        _read(root / "cases" / (c["id"] + ".json"))
        for c in declared
        if (root / "cases" / (c["id"] + ".json")).exists()
    ]
    scored = [r for r in rows if "error_pct" in r]
    domain = [r for r in scored if not r["ood_reasons"]]
    mean = lambda xs: sum(abs(r["error_pct"]) for r in xs) / len(xs) if xs else None
    report = dict(
        expected=len(declared),
        attempted=len(rows),
        scored=len(scored),
        in_domain=len(domain),
        diagnostic_mape_pct=mean(scored),
        full_254_mape_pct=mean(scored) if len(scored) == 254 else None,
        in_domain_mape_pct=mean(domain),
        by_operator={
            op: dict(
                count=len(rr := [r for r in scored if r["case"]["op"] == op]),
                mape_pct=mean(rr),
            )
            for op in sorted({c["op"] for c in declared})
        },
        failures=[
            dict(case=r["case"], error=r.get("error"))
            for r in rows
            if "error_pct" not in r
        ],
    )
    _write(root / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tilebench-dir", type=Path, required=True)
    from microbench.gpu.common.cases import formal_holdout_splits

    parser.add_argument("--splits", type=Path, default=formal_holdout_splits())
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--supplement-only",
        action="store_true",
        help="Supplement matmul, attention and BF16 mul2 compatibility cases",
    )
    args = parser.parse_args()
    import torch
    from triton_viz.performance.gpu import predict
    from triton_viz.performance.triton_observe import observe
    from microbench.gpu.harness.measure import snapshot, assert_available, measure

    declared = cases(args.splits)
    assert len(declared) == 254
    model = _read(args.calibration)
    digest = model.pop("digest")
    if stable_digest(model) != digest:
        raise ValueError("Frozen model digest mismatch")
    old = _read(args.calibration_manifest)
    baseline = snapshot()
    allowed = tuple(p["pid"] for p in baseline["graphics_processes"])
    for _ in range(3):
        assert_available(snapshot(), allowed_graphics=allowed)
        time.sleep(0.2)
    torch.cuda.set_device(0)
    identity = _identity(0)
    # Source snapshot necessarily changes to add this evaluator; hardware and
    # runtime must not. Preserve both identities rather than forging a new fit.
    for key in old["identity"]:
        if (
            key in identity
            and key != "source_digest"
            and old["identity"][key] != identity[key]
        ):
            raise ValueError(f"Calibration environment mismatch: {key}")
    manifest = dict(
        supplement_only=args.supplement_only,
        cases=declared,
        identity=identity,
        calibration_digest=digest,
        calibration_fingerprint=model["fingerprint"],
        tilebench_sources={
            str(p.relative_to(args.tilebench_dir)): __import__("hashlib")
            .sha256(p.read_bytes())
            .hexdigest()
            for p in args.tilebench_dir.glob("*/impl_triton.py")
        },
    )
    manifest_path = args.root / "manifest.json"
    if manifest_path.exists() and _read(manifest_path) != manifest:
        raise ValueError("Resume manifest mismatch")
    _write(manifest_path, manifest)
    selected = (
        [
            c
            for c in declared
            if c["op"] in {"tiled_attention", "matmul_fp32_fp16_fp8"}
            or (c["op"] == "mul2" and c["dtype"] == "bfloat16")
        ]
        if args.supplement_only
        else declared
    )
    for case in selected[: args.limit]:
        path = args.root / "cases" / (case["id"] + ".json")
        if path.exists():
            continue
        row = dict(case=case)
        try:
            assert_available(snapshot(), own_pid=os.getpid(), allowed_graphics=allowed)
            (kernel, grid, cpu_args, kwargs), output, reference = prepare(
                case, args.tilebench_dir, adapters=args.supplement_only
            )
            row["implementation"] = (
                "explicit_semantic_adapter"
                if args.supplement_only
                and (
                    case["op"] == "tiled_attention"
                    or (
                        case["op"] == "matmul_fp32_fp16_fp8"
                        and case["dtype"] == "bfloat16"
                    )
                )
                else "unmodified_tilebench"
            )
            try:
                source = observe(kernel, grid, *cpu_args, **kwargs)
                _write(args.root / "sources" / (case["id"] + ".json"), source)
                prediction = predict(
                    source,
                    model,
                    fingerprint=model["fingerprint"],
                    sm_count=identity["sm_count"],
                    strict=False,
                )
                row.update(
                    predicted_us=prediction["latency_us"],
                    ood_reasons=prediction["ood_reasons"],
                )
            except Exception as exc:
                row["prediction_error"] = f"{type(exc).__name__}: {exc}"
            gpu_args = tuple(
                x.to("cuda") if isinstance(x, torch.Tensor) else x for x in cpu_args
            )

            def launch():
                kernel[grid](*gpu_args, **kwargs)

            launch()
            # Output can be a view of a launch argument; locate its shared storage.
            out_index = next(
                i
                for i, x in enumerate(cpu_args)
                if isinstance(x, torch.Tensor)
                and x.untyped_storage().data_ptr()
                == output.untyped_storage().data_ptr()
            )
            actual = gpu_args[out_index].reshape(output.shape).cpu()
            if case["op"] == "matmul_fp32_fp16_fp8" and case["dtype"] == "float32":
                # Tilebench explicitly requests TF32. Compare against a forward
                # error bound for 10-bit input mantissas and FP32 accumulation,
                # rather than an FP32 elementwise tolerance near cancellation.
                unit = 2.0**-10
                gamma = case["cols"] * 2.0**-24 / (1 - case["cols"] * 2.0**-24)
                bound = (2 * unit + unit * unit + gamma) * (
                    cpu_args[0].abs() @ cpu_args[1].abs()
                )
                residual = (actual - reference).abs()
                if (
                    not torch.isfinite(actual).all()
                    or not (residual <= bound + 1e-6).all()
                ):
                    raise ValueError(
                        "TF32 output exceeds input-quantization/accumulation bound"
                    )
                row["correctness"] = dict(
                    protocol="tf32_forward_error_bound",
                    max_absolute_error=float(residual.max()),
                    relative_l2_error=float(
                        torch.linalg.vector_norm(residual)
                        / torch.linalg.vector_norm(reference)
                    ),
                )
            else:
                torch.testing.assert_close(
                    actual,
                    reference,
                    rtol=0.03 if case["dtype"] == "bfloat16" else 0.01,
                    atol=0.02 if case["dtype"] == "bfloat16" else 0.002,
                )
            for attempt in range(3):
                measurement = measure(launch, allowed_graphics=allowed)
                _write(
                    args.root / "attempts" / f"{case['id']}_{attempt}.json", measurement
                )
                if not measurement["contaminated"]:
                    break
            if measurement["contaminated"]:
                raise ValueError("All timing batches contaminated")
            row["measured_us"] = measurement["latency_us"]
            if "predicted_us" in row:
                row["error_pct"] = 100 * (row["predicted_us"] / row["measured_us"] - 1)
            else:
                row["error"] = row["prediction_error"]
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        _write(path, row)
        print(
            case["id"],
            {k: v for k, v in row.items() if k not in {"case", "ood_reasons"}},
            flush=True,
        )
        summarize(args.root, declared)
    print(json.dumps(summarize(args.root, declared), indent=2), flush=True)


if __name__ == "__main__":
    main()
