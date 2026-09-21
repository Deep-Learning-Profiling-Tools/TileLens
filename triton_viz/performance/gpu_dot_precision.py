"""Source-semantic dot pricing features, independent of target compilation."""

from collections import defaultdict

import numpy as np

DOT_CLASSES = ("ieee_fp32", "tf32", "bf16", "fp16")
DOT_FEATURES = tuple(
    f"{prefix}_{kind}"
    for prefix in ("dot_flops", "program_dot_p90")
    for kind in DOT_CLASSES
)


WAVE_DOT_FEATURES = tuple(f"wave_dot_flops_{kind}" for kind in DOT_CLASSES)


def wave_dot_features(features):
    """Source program-tail demand times hardware SM waves, not measured occupancy."""
    return {
        f"wave_dot_flops_{kind}": features["waves"]
        * features[f"program_dot_p90_{kind}"]
        for kind in DOT_CLASSES
    }


def dot_features(source):
    """Return split work, explicit semantic configurations and OOD reasons.

    The spelling 'ieee' on FP16/BF16 does not turn its input into FP32. An
    output/epilogue cast is not an input dtype. Unknown traces are never guessed.
    Each program distribution includes zero demand from programs of other kinds.
    """
    totals = dict.fromkeys(DOT_CLASSES, 0.0)
    programs = defaultdict(lambda: dict.fromkeys(DOT_CLASSES, 0.0))
    configurations = set()
    reasons = set()
    for event in source["events"]:
        program = programs[tuple(event["program"])]
        if event["op"] != "dot":
            continue
        inputs = tuple(event.get("dot_input_dtypes", ()))
        accumulator = event.get("dot_accumulator_dtype")
        precision = event.get("dot_input_precision")
        if len(inputs) != 2 or accumulator is None or precision is None:
            reasons.add("missing_dot_precision_metadata")
            continue
        kind = None
        if inputs == ("fp32", "fp32") and precision in {"ieee", "tf32"}:
            kind = "ieee_fp32" if precision == "ieee" else "tf32"
        elif inputs == ("bf16", "bf16"):
            kind = "bf16"
        elif inputs == ("fp16", "fp16"):
            kind = "fp16"
        if kind is None or accumulator != "fp32":
            reasons.add(
                "unsupported_dot_precision:"
                + "/".join((*inputs, str(accumulator), str(precision)))
            )
            continue
        shapes = event["input_shapes"]
        if (
            len(shapes) < 2
            or len(shapes[0]) != 2
            or len(shapes[1]) != 2
            or shapes[0][1] != shapes[1][0]
        ):
            reasons.add("unsupported_dot_geometry")
            continue
        flops = 2 * shapes[0][0] * shapes[0][1] * shapes[1][1]
        totals[kind] += flops
        program[kind] += flops
        configurations.add((kind, *inputs, accumulator, precision))
    if len(programs) != source["program_count"] or not programs:
        raise ValueError("Dot distributions require a complete nonempty program trace")
    features = {f"dot_flops_{kind}": value for kind, value in totals.items()}
    features.update(
        {
            f"program_dot_p90_{kind}": float(
                np.quantile([p[kind] for p in programs.values()], 0.9)
            )
            for kind in DOT_CLASSES
        }
    )
    return features, [list(c) for c in sorted(configurations)], sorted(reasons)
