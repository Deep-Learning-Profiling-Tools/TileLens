"""Role-separated experiment declarations; importing this module needs no GPU."""

import json
from itertools import product
from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def load_cases(suite, role):
    if suite not in {"pilot", "compositional", "coverage"}:
        raise ValueError("Unknown GPU control suite")
    if role not in {"control", "holdout"}:
        raise ValueError("Unknown artifact role")
    # Only the requested role's declaration is opened. Tilebench254 cannot be
    # selected as a control suite or accidentally included in these controls.
    data = json.loads((CONFIGS / f"{suite}_{role}.json").read_text())
    if (data["schema"], data["suite"], data["role"]) != (
        "triton-viz.gpu-cases.v1",
        suite,
        role,
    ):
        raise ValueError("Case declaration identity mismatch")
    cases = data["cases"]
    if suite == "coverage" and role == "control":
        cases = cases + paired_controls()
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    return cases


def paired_controls():
    """Expand the predeclared control-only matrix; keep paired cases in one CV group."""
    data = json.loads((CONFIGS / "serial_parallel_control.json").read_text())
    if (
        data["schema"] != "triton-viz.gpu-paired-controls.v1"
        or data["role"] != "control"
    ):
        raise ValueError("Invalid paired control declaration")
    result = []
    for block, depth, dtype, op in product(
        data["blocks"], data["depths"], data["dtypes"], data["operations"]
    ):
        pair = f"paired_{op}_{dtype}_b{block}_d{depth}"
        for topology in data["topologies"]:
            result.append(
                dict(
                    id=f"{pair}_{topology}",
                    pair_id=pair,
                    kind="coverage_paired",
                    programs=data["programs"],
                    block=block,
                    repeat=depth,
                    dtype=dtype,
                    operation=op,
                    topology=topology,
                    num_warps=data["num_warps"],
                    num_stages=data["num_stages"],
                    cv_group=f"paired_b{block}_d{depth}",
                )
            )
    return result


def formal_holdout_splits():
    declaration = json.loads((CONFIGS / "tilebench254_holdout.json").read_text())
    if declaration["role"] != "holdout":
        raise ValueError("254-point suite must remain holdout-only")
    return CONFIGS.parents[1] / declaration["source"]
