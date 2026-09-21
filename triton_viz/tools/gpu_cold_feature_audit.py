"""Control-only identifiability audit; never admit or fit cold latency data.

Equal pricing vectors must produce equal deterministic predictions. An oracle
within each equal-vector group gives a lower bound on empirical MAPE, including
measurement noise. Removing collisions does not establish generalization.
"""

import argparse
import math
from pathlib import Path

from triton_viz.performance.gpu_distributions import FEATURE_SETS
from triton_viz.performance.gpu_dot_precision import dot_features, wave_dot_features
from triton_viz.performance.gpu_memory import memory_features
from triton_viz.tools.gpu_cold_control_audit import audit
from triton_viz.tools.gpu_cost_model_pipeline import _read, _write
from triton_viz.tools.gpu_distribution_experiments import enrich


def collision_bound(rows, keys):
    groups = {}
    for row in rows:
        latency = row["latency_us"]
        vector = tuple(row["features"][key] for key in keys)
        if (
            not math.isfinite(latency)
            or latency <= 0
            or any(not math.isfinite(v) for v in vector)
        ):
            raise ValueError("Require finite features and positive measurements")
        groups.setdefault(vector, []).append(row)
    if not rows:
        raise ValueError("Require nonempty controls")
    error, collisions = 0.0, []
    for group in groups.values():
        values = sorted(r["latency_us"] for r in group)
        halfway = sum(1 / v for v in values) / 2
        weight = 0
        for center in values:
            weight += 1 / center
            if weight >= halfway:
                break
        group_error = sum(abs(center - v) / v for v in values)
        error += group_error
        if len(group) > 1:
            collisions.append(
                dict(
                    members=[
                        dict(id=r["id"], latency_us=r["latency_us"]) for r in group
                    ],
                    count=len(group),
                    latency_ratio=max(values) / min(values),
                    oracle_mape_pct=100 * group_error / len(group),
                )
            )
    return dict(
        features=list(keys),
        count=len(rows),
        distinct_vectors=len(groups),
        oracle_mape_floor_pct=100 * error / len(rows),
        collisions=sorted(collisions, key=lambda g: g["latency_ratio"], reverse=True),
    )


def run(roots, sm_count):
    if sm_count <= 0:
        raise ValueError("Require hardware SM count")
    rows, protocol = [], None
    for root in roots:
        integrity = audit(root)
        if not integrity["measurement_integrity_passed"]:
            raise ValueError("Incomplete cold control collection; no point removal")
        manifest = _read(root / "manifest.json")
        identity = {
            k: manifest[k]
            for k in (
                "metric",
                "timestamp_method",
                "launch_mode",
                "library_sha256",
                "kernels_per_sample",
                "packages",
            )
        }
        if protocol is not None and protocol != identity:
            raise ValueError("Do not mix measurement protocols")
        protocol = identity
        for case in manifest["cases"]:
            raw = _read(root / "controls" / (case["id"] + ".json"))
            source = raw["source"]
            features, _, _ = enrich(source, sm_count)
            dot, _, dot_reasons = dot_features(source)
            memory, memory_reasons = memory_features(source)
            if dot_reasons or memory_reasons:
                raise ValueError(
                    f"Unsupported control retained, audit cannot complete: {case['id']}"
                )
            features.update(dot)
            features.update(memory)
            features.update(wave_dot_features(features))
            features.update(
                source_num_warps=source["num_warps"],
                source_num_stages=source["num_stages"],
            )
            rows.append(
                dict(id=case["id"], latency_us=raw["median_us"], features=features)
            )
        print(
            f"Audited and observed {root.name}: {len(manifest['cases'])} controls",
            flush=True,
        )
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate controls across roots")
    comparisons = {}
    for name in (
        "dot_precision_combined",
        "dot_precision_memory",
        "dot_precision_memory_wave",
    ):
        keys = FEATURE_SETS[name]
        comparisons[name] = collision_bound(rows, keys)
        comparisons[name + "_source_launch_configuration"] = collision_bound(
            rows, (*keys, "source_num_warps", "source_num_stages")
        )
    return dict(
        role="control",
        eligible_for_fit=False,
        released_model=None,
        count=len(rows),
        protocol=protocol,
        sm_count=sm_count,
        comparisons=comparisons,
        caveat="Unadmitted cold measurements, all controls retained. Oracle feature-collision bound only; no latency fitting, CV gate, or holdout evaluation.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--sm-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Require fresh diagnostic output")
    result = run(args.roots, args.sm_count)
    _write(args.output, result)
    print({k: v["oracle_mape_floor_pct"] for k, v in result["comparisons"].items()})


if __name__ == "__main__":
    main()
