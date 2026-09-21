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
        "pressure_pipeline",
        "resource_transfer",
        "composition_component",
    }:
        raise ValueError("Unknown GPU control suite")
    if role not in {"control", "holdout"}:
        raise ValueError("Unknown artifact role")
    if suite == "pressure_pipeline":
        if role == "holdout":
            return []
        data = json.loads((CONFIGS / "pressure_pipeline_control.json").read_text())
        if (
            data["schema"],
            data["role"],
            data["template_suite"],
            data["num_stages"],
        ) != ("triton-viz.gpu-pressure-pipeline-controls.v1", "control", "pressure", 2):
            raise ValueError("Invalid matched pressure pipeline declaration")
        templates = load_cases("pressure", "control")
        if any(case["num_stages"] != 1 for case in templates):
            raise ValueError("Matched pipeline controls require stage-1 templates")
        return [
            {**case, "id": case["id"] + "_s2", "num_stages": 2} for case in templates
        ]
    if suite == "composition_component":
        if role == "holdout":
            return []
        data = json.loads((CONFIGS / "composition_component_control.json").read_text())
        if (data["schema"], data["role"]) != (
            "triton-viz.gpu-composition-component-controls.v1",
            "control",
        ):
            raise ValueError("Invalid composition component declaration")
        return [
            dict(
                id=f"component_{dtype}_{precision}_{bm}x{bn}x{bk}_w{warps}_s{stages}",
                kind="geometry_dot",
                programs=data["programs"],
                dtype=dtype,
                precision=precision,
                bm=bm,
                bn=bn,
                bk=bk,
                num_warps=warps,
                num_stages=stages,
                repeat=data["repeat"],
                reuse="none",
                cv_group=f"resource_composition_{bm}x{bn}x32",
            )
            for (dtype, precision), warps, stages, (bm, bn, bk) in product(
                data["precisions"], data["warps"], data["stages"], data["tiles"]
            )
        ]
    if suite == "resource_transfer":
        if role == "holdout":
            return []
        data = json.loads((CONFIGS / "resource_transfer_control.json").read_text())
        if (data["schema"], data["role"]) != (
            "triton-viz.gpu-resource-transfer-controls.v1",
            "control",
        ):
            raise ValueError("Invalid resource transfer declaration")
        result = []
        for family in ("dot", "composition"):
            variants = (
                data["dot_repeats"] if family == "dot" else data["composition_modes"]
            )
            for (dtype, precision), warps, stages, (bm, bn, bk), variant in product(
                data["precisions"],
                data["warps"],
                data["stages"],
                data[f"{family}_tiles"],
                variants,
            ):
                result.append(
                    dict(
                        id=f"resource_{family}_{dtype}_{precision}_{bm}x{bn}x{bk}_w{warps}_s{stages}_{variant}",
                        kind="geometry_dot"
                        if family == "dot"
                        else "structure_composition",
                        programs=data["programs"],
                        dtype=dtype,
                        precision=precision,
                        bm=bm,
                        bn=bn,
                        bk=bk,
                        num_warps=warps,
                        num_stages=stages,
                        repeat=variant if family == "dot" else 1,
                        **(
                            {"reuse": "none"}
                            if family == "dot"
                            else {"variant": variant}
                        ),
                        cv_group=f"resource_{family}_{bm}x{bn}x{bk}",
                    )
                )
        return result
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
