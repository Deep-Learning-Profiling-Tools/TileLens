"""Source-only program distributions inspired by Concorde (arXiv:2503.23076).

These are structural workload/latency proxies, not Concorde's CPU throughput
bounds. No target instruction traces or calibrated physical latencies are read.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .gpu import FEATURES, _ALU, _REDUCE, _SFU

PROGRAM_FEATURES = (
    "program_alu_p90",
    "program_sfu_p90",
    "program_reduction_p90",
    "program_tensor_p90",
)
PATH_FEATURES = ("path_alu_p90", "path_sfu_p90", "path_reduction_p90")
FEATURE_SETS = {
    "aggregate": FEATURES,
    "program_distribution": FEATURES + PROGRAM_FEATURES,
    "dependency_distribution": FEATURES + PATH_FEATURES,
    "combined_distribution": FEATURES + PROGRAM_FEATURES + PATH_FEATURES,
}


def distribution_features(source):
    """Summarize per-program demand and longest exposed component paths.

    Layout operations carry predecessor path state at zero cost. Each component
    is analyzed independently; its longest path can differ from other components.
    Older v1 traces may lack store-value and dot-accumulator edges; recollect
    those traces to include the complete observed operand dependencies.
    """
    programs = defaultdict(
        lambda: dict.fromkeys(("alu", "sfu", "reduction", "tensor"), 0.0)
    )
    critical = defaultdict(lambda: dict.fromkeys(("alu", "sfu", "reduction"), 0.0))
    paths = {}
    event_programs = {}
    for event in source["events"]:
        seq = event["seq"]
        program = tuple(event["program"])
        if seq in paths:
            raise ValueError("Duplicate source event sequence")
        demand = programs[program]
        longest = critical[program]
        prior = []
        for dependency in event["dependencies"]:
            if dependency not in paths:
                raise ValueError("Missing or non-topological source dependency")
            if event_programs[dependency] != program:
                raise ValueError("Cross-program dependencies are not supported")
            prior.append(paths[dependency])
        state = {
            component: max((p[component] for p in prior), default=0.0)
            for component in longest
        }
        op = event["op"]
        dtype = event["dtype"]
        floating = "pointer" not in dtype and any(
            t in dtype for t in ("float", "fp16", "fp32", "fp64", "bf16")
        )
        if op in _REDUCE:
            count = math.prod(event["input_shapes"][0])
            width = max(1, count // max(1, event["elements"]))
            steps = math.ceil(math.log2(width))
            demand["reduction"] += steps
            state["reduction"] += steps
        elif op == "dot":
            a, b = event["input_shapes"][:2]
            if len(a) != 2 or len(b) != 2 or a[1] != b[0]:
                raise ValueError("Unsupported dot geometry")
            demand["tensor"] += 2 * a[0] * a[1] * b[1]
        elif floating and op in {"binary_op", "unary_op", "fma", "rsqrt", "fabs"}:
            primitive = event.get("primitive", op)
            component = (
                "sfu"
                if primitive in _SFU
                else "alu"
                if primitive in _ALU | {"fma", "fabs"}
                else None
            )
            if component:
                demand[component] += math.ceil(event["elements"] / 32)
                state[component] += 1
        paths[seq] = state
        event_programs[seq] = program
        for component in longest:
            longest[component] = max(longest[component], state[component])
    if not programs or len(programs) != source["program_count"]:
        raise ValueError(
            "Program distributions require a complete, nonempty launch trace"
        )
    features = {}
    distributions = {}
    for prefix, values in (("program", programs), ("path", critical)):
        for component in next(iter(values.values())):
            samples = [row[component] for row in values.values()]
            summary = {
                "mean": float(np.mean(samples)),
                "p50": float(np.quantile(samples, 0.5)),
                "p90": float(np.quantile(samples, 0.9)),
                "max": float(np.max(samples)),
            }
            distributions[f"{prefix}_{component}"] = summary
            features[f"{prefix}_{component}_p90"] = summary["p90"]
    return features, distributions
