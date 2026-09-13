"""Role-separated experiment declarations; importing this module needs no GPU."""

import json
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
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    return cases


def formal_holdout_splits():
    declaration = json.loads((CONFIGS / "tilebench254_holdout.json").read_text())
    if declaration["role"] != "holdout":
        raise ValueError("254-point suite must remain holdout-only")
    return CONFIGS.parents[1] / declaration["source"]
