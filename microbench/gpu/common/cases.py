"""Role-separated experiment declarations; importing this module needs no GPU."""

import json
from itertools import product
from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def load_cases(suite, role):
    if suite not in {
        "pilot",
        "compositional",
        "coverage",
        "precision",
        "geometry",
        "structure",
        "stability",
        "pressure",
    }:
        raise ValueError("Unknown GPU control suite")
    if role not in {"control", "holdout"}:
        raise ValueError("Unknown artifact role")
    if suite == "pressure":
        if role == "holdout":
            return []
        data = json.loads((CONFIGS / "pressure_control.json").read_text())
        if (data["schema"], data["role"]) != (
            "triton-viz.gpu-register-pressure-controls.v1",
            "control",
        ):
            raise ValueError("Invalid register pressure declaration")
        return [
            dict(
                id=f"pressure_p{p}_{dtype}_{precision}_{bm}x{bn}x{bk}_w{warps}",
                kind="geometry_dot",
                programs=p,
                dtype=dtype,
                precision=precision,
                bm=bm,
                bn=bn,
                bk=bk,
                num_warps=warps,
                num_stages=data["num_stages"],
                repeat=data["repeat"],
                reuse="none",
                cv_group=f"pressure_{bm}x{bn}x{bk}",
            )
            for p, (dtype, precision), (bm, bn, bk), warps in product(
                data["programs"], data["precisions"], data["tiles"], data["warps"]
            )
        ]
    if suite == "stability":
        if role == "holdout":
            return []
        data = json.loads((CONFIGS / "stability_control.json").read_text())
        if (data["schema"], data["role"]) != (
            "triton-viz.gpu-selection-stability-controls.v1",
            "control",
        ):
            raise ValueError("Invalid selection stability declaration")
        templates = [
            c
            for base in data["template_suites"]
            for c in load_cases(base, "control")
            if c.get("programs") == data["template_programs"]
        ]
        if len({c["id"] for c in templates}) != len(templates):
            raise ValueError("Duplicate stability templates")
        result = []
        for programs, template in product(data["programs"], templates):
            case = {
                **template,
                "id": f"stability_p{programs}__{template['id']}",
                "programs": programs,
                "cv_group": str(programs),
                "template_id": template["id"],
            }
            if "pair_id" in template:
                case["pair_id"] = f"stability_p{programs}__{template['pair_id']}"
            result.append(case)
        return result
    if suite == "structure":
        if role == "holdout":
            return []
        data = json.loads((CONFIGS / "structure_control.json").read_text())
        if (data["schema"], data["role"]) != (
            "triton-viz.gpu-dot-structure-controls.v1",
            "control",
        ):
            raise ValueError("Invalid dot structure declaration")
        result = []
        for family in ("stream", "composition"):
            for p, (dtype, precision), (bm, bn, bk), variant in product(
                data["programs"],
                data["precisions"],
                data[f"{family}_tiles"],
                data["stream_layouts"]
                if family == "stream"
                else data["composition_modes"],
            ):
                group = f"structure_{family}_p{p}_{dtype}_{precision}_{bm}x{bn}x{bk}"
                result.append(
                    dict(
                        id=f"{group}_{variant}",
                        kind=f"structure_{family}",
                        programs=p,
                        dtype=dtype,
                        precision=precision,
                        bm=bm,
                        bn=bn,
                        bk=bk,
                        variant=variant,
                        repeat=data["stream_repeat"] if family == "stream" else 1,
                        num_warps=data["num_warps"],
                        num_stages=data["num_stages"],
                        cv_group=str(p),
                        pair_id=group,
                    )
                )
        return result
    if suite == "geometry":
        # A diagnostic, control-only factorial experiment; no target declaration.
        if role == "holdout":
            return []
        data = json.loads((CONFIGS / "geometry_control.json").read_text())
        if (data["schema"], data["role"]) != (
            "triton-viz.gpu-dot-geometry-controls.v1",
            "control",
        ):
            raise ValueError("Invalid dot geometry controls")
        result = []
        for p, (dtype, precision), (bm, bn, bk), reuse, stages in product(
            data["programs"],
            data["precisions"],
            data["tiles"],
            data["reuse"],
            data["stages"],
        ):
            group = f"geometry_p{p}_{dtype}_{precision}_{bm}x{bn}x{bk}"
            result.append(
                dict(
                    id=f"{group}_{reuse}_s{stages}",
                    kind="geometry_dot",
                    programs=p,
                    dtype=dtype,
                    precision=precision,
                    bm=bm,
                    bn=bn,
                    bk=bk,
                    reuse=reuse,
                    num_stages=stages,
                    repeat=data["repeat"],
                    num_warps=data["num_warps"],
                    cv_group=group,
                )
            )
        return result
    if suite == "precision":
        base = load_cases("coverage", role)
        if role == "holdout":
            return base
        data = json.loads((CONFIGS / "precision_control.json").read_text())
        if (
            data["role"] != "control"
            or data["schema"] != "triton-viz.gpu-dot-precision-controls.v1"
        ):
            raise ValueError("Invalid dot precision controls")
        extra = []
        for programs, repeat, (dtype, precision) in product(
            data["programs"], data["repeats"], data["precisions"]
        ):
            extra.append(
                dict(
                    id=f"precision_dot_p{programs}_{dtype}_{precision}_r{repeat}",
                    kind="coverage_dot",
                    programs=programs,
                    repeat=repeat,
                    dtype=dtype,
                    precision=precision,
                    cv_group=str(programs),
                    **{
                        k: data[k]
                        for k in ("bm", "bn", "bk", "num_warps", "num_stages")
                    },
                )
            )
        return base + extra
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
