"""Offline control-only packing/loop interventions; never launches or fits kernels."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from triton_viz.tools.gpu_cost_model_pipeline import _write


def scalarize(ptx):
    """Split unpredicated register-only rn f32x2 FMA, preserving both lanes.

    PTX mov.b64 unpack/pack and scalar fma.rn.f32 keep the lane arithmetic.
    This is not general vectorization removal or GPU numerical validation.
    """
    if "%tv_scalar" in ptx:
        raise ValueError("Intervention register namespace already present")
    pattern = re.compile(
        r"^\s*fma\.rn\.f32x2\s+(%\w+),\s*(%\w+),\s*(%\w+),\s*(%\w+);\s*$", re.M
    )

    def replace(match):
        d, a, b, c = match.groups()
        return "\n".join(
            [
                "{",
                ".reg .b32 %tv_scalar<8>;",
                f"mov.b64 {{%tv_scalar0, %tv_scalar1}}, {a};",
                f"mov.b64 {{%tv_scalar2, %tv_scalar3}}, {b};",
                f"mov.b64 {{%tv_scalar4, %tv_scalar5}}, {c};",
                "fma.rn.f32 %tv_scalar6, %tv_scalar0, %tv_scalar2, %tv_scalar4;",
                "fma.rn.f32 %tv_scalar7, %tv_scalar1, %tv_scalar3, %tv_scalar5;",
                f"mov.b64 {d}, {{%tv_scalar6, %tv_scalar7}};",
                "}",
            ]
        )

    rewritten, count = pattern.subn(replace, ptx)
    if re.search(r"\bfma\.[^\s;]*f32x2", rewritten):
        raise ValueError("Unsupported predicated/modified/non-register packed FMA")
    return rewritten, count


def scalarize_values(ptx):
    """Separate packed data-value registers, retaining shared-load byte width.

    Only a closed mov/FMA/shared-load dataflow is accepted. Pointer arithmetic,
    predicates and other operations touching a selected value fail closed.
    """
    if "%tv_pair" in ptx:
        raise ValueError("Intervention register namespace already present")
    fma = re.compile(r"fma\.rn\.f32x2\s+(%rd\d+),\s*(%rd\d+),\s*(%rd\d+),\s*(%rd\d+);")
    matches = list(fma.finditer(ptx))
    if not matches:
        if "f32x2" in ptx:
            raise ValueError("Unsupported packed operation")
        return ptx, 0
    selected = {r for m in matches for r in m.groups()}
    copies = re.findall(r"mov\.b64\s+(%rd\d+),\s*(%rd\d+);", ptx)
    loads = re.findall(r"ld\.shared\.v2\.b64\s+\{(%rd\d+),\s*(%rd\d+)\}", ptx)
    while True:
        expanded = selected | {
            r for pair in copies + loads if selected.intersection(pair) for r in pair
        }
        if expanded == selected:
            break
        selected = expanded
    bound = max(int(r[3:]) for r in selected) + 1
    if len(re.findall(r"\.visible\s+\.entry\b", ptx)) != 1:
        raise ValueError("Require a single control entry")

    def lane(reg, part):
        return f"%tv_pair_{part}{reg[3:]}"

    output = []
    for line in ptx.splitlines():
        if not selected.intersection(re.findall(r"%rd\d+\b", line)):
            output.append(line)
            continue
        instruction = line.strip()
        m = fma.fullmatch(instruction)
        if m:
            output.extend(
                "fma.rn.f32 " + ", ".join(lane(r, part) for r in m.groups()) + ";"
                for part in ("lo", "hi")
            )
            continue
        m = re.fullmatch(
            r"mov\.b64\s+(%rd\d+),\s*\{([^,{}]+),\s*([^,{}]+)\};", instruction
        )
        if m:
            output.extend(
                f"mov.b32 {lane(m[1], part)}, {value.strip()};"
                for part, value in zip(("lo", "hi"), m.groups()[1:])
            )
            continue
        m = re.fullmatch(
            r"mov\.b64\s+\{([^,{}]+),\s*([^,{}]+)\},\s*(%rd\d+);", instruction
        )
        if m:
            output.extend(
                f"mov.b32 {value.strip()}, {lane(m[3], part)};"
                for part, value in zip(("lo", "hi"), m.groups()[:2])
            )
            continue
        m = re.fullmatch(r"mov\.b64\s+(%rd\d+),\s*(%rd\d+);", instruction)
        if m:
            output.extend(
                f"mov.b32 {lane(m[1], part)}, {lane(m[2], part)};"
                for part in ("lo", "hi")
            )
            continue
        m = re.fullmatch(
            r"ld\.shared\.v2\.b64\s+\{(%rd\d+),\s*(%rd\d+)\},\s*(\[[^\]]+\]);",
            instruction,
        )
        if m:
            registers = ", ".join(
                lane(r, part) for r in m.groups()[:2] for part in ("lo", "hi")
            )
            output.append(f"ld.shared.v4.b32 {{{registers}}}, {m[3]};")
            continue
        raise ValueError(f"Unsupported operation touching packed data: {instruction}")
    rewritten = "\n".join(output) + "\n"
    first_decl = re.search(r"^\s*\.reg\b", rewritten, re.M)
    if first_decl is None:
        raise ValueError("Missing control register declarations")
    rewritten = (
        rewritten[: first_decl.start()]
        + f"\n.reg .b32 %tv_pair_lo<{bound}>;\n.reg .b32 %tv_pair_hi<{bound}>;\n"
        + rewritten[first_decl.start() :]
    )
    if "f32x2" in rewritten:
        raise ValueError("Unhandled packed operation remains")
    return rewritten, len(matches)


def unroll_control_loop(ptx, trips):
    """Duplicate one independently checked uniform affine control loop."""
    if isinstance(trips, bool) or not isinstance(trips, int) or trips < 1:
        raise ValueError("Require positive declared trips")
    branches = list(re.finditer(r"^[ \t]*@(%p\d+)\s+bra\s+(\$\w+);[ \t]*$", ptx, re.M))
    if not branches and trips == 1 and not re.search(r"\bbra\b", ptx):
        return ptx, 0
    if len(branches) != 1 or len(re.findall(r"\bbra\b", ptx)) != 1:
        raise ValueError("Require one conditional backedge")
    branch = branches[0]
    pred, label = branch.groups()
    headers = list(re.finditer(r"^" + re.escape(label) + r":[^\n]*\n", ptx, re.M))
    if len(headers) != 1 or headers[0].end() >= branch.start():
        raise ValueError("Invalid loop header")
    header = headers[0]
    body = ptx[header.end() : branch.start()]
    if re.search(r"^\s*\$\w+:|\b(?:call|brx|ret)\b|\.reg\b|\.local\b", body, re.M):
        raise ValueError("Unsupported internal loop flow or declarations")
    comparisons = re.findall(
        r"setp\.ne\.b64\s+" + re.escape(pred) + r",\s*(%rd\d+),\s*(\d+);", body
    )
    if len(comparisons) != 1:
        raise ValueError("Require a uniform integer loop test")
    induction, bound = comparisons[0]
    predicate_lines = [
        line.strip()
        for line in body.splitlines()
        if re.search(re.escape(pred) + r"\b", line)
    ]
    if len(predicate_lines) != 1 or not predicate_lines[0].startswith("setp.ne.b64"):
        raise ValueError("Loop predicate must have one unpredicated definition")
    increments = re.findall(
        r"add\.s64\s+"
        + re.escape(induction)
        + r",\s*"
        + re.escape(induction)
        + r",\s*(\d+);",
        body,
    )
    initialization = re.findall(
        r"mov\.b64\s+" + re.escape(induction) + r",\s*0;", ptx[: header.start()]
    )
    writes = re.findall(r"^\s*([\w.]+)\s+" + re.escape(induction) + r",", ptx, re.M)
    if (
        len(increments) != 1
        or len(initialization) != 1
        or writes != ["mov.b64", "add.s64"]
        or int(increments[0]) <= 0
        or int(bound) != trips * int(increments[0])
        or int(bound) >= 2**63
    ):
        raise ValueError("Declared trips do not match the affine control induction")
    increment = re.search(r"add\.s64\s+" + re.escape(induction) + r",", body)
    if increment.start() >= body.index(predicate_lines[0]):
        raise ValueError("Require post-increment loop comparison")
    return ptx[: header.end()] + body * trips + ptx[branch.end() :], trips


def variants(row, mode="fma"):
    if row.get("role") != "control":
        raise ValueError("Only declared control artifacts may be transformed")
    ptx = row["artifacts"]["ptx"]
    if hashlib.sha256(ptx.encode()).hexdigest() != row["artifact_sha256"]["ptx"]:
        raise ValueError("Control PTX digest mismatch")
    if mode not in {"fma", "value_pairs", "unroll"}:
        raise ValueError("Unknown intervention")
    if mode == "unroll":
        rewritten, count = unroll_control_loop(ptx, row["case"]["repeat"])
        return {"original": ptx, "unrolled": rewritten}, count
    rewritten, count = (scalarize if mode == "fma" else scalarize_values)(ptx)
    return {"original": ptx, "scalar_" + mode: rewritten}, count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--ptxas", type=Path, required=True)
    parser.add_argument("--cuobjdump", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regalloc-opt-level", type=int, choices=(0, 1, 2))
    parser.add_argument(
        "--mode", choices=("fma", "value_pairs", "unroll"), default="fma"
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Use a fresh experiment directory")
    manifest = json.loads((args.resource_root / "manifest.json").read_text())
    if manifest.get("role") != "control":
        raise ValueError("Require a control manifest")
    ids = [
        f"resource_dot_float32_ieee_128x256x32_w{w}_s{s}_{r}"
        for w, s, r in ((4, 1, 1), (4, 1, 5), (4, 2, 5), (8, 1, 5))
    ]
    cases = {c["id"]: c for c in manifest["cases"]}
    if not set(ids) <= set(cases):
        raise ValueError("Missing declared matched controls")
    version = subprocess.run(
        [str(args.ptxas), "--version"], check=True, capture_output=True, text=True
    ).stdout
    _write(
        args.output / "manifest.json",
        dict(
            role="control",
            cases=[cases[i] for i in ids],
            ptxas_version=version,
            ptxas_sha256=hashlib.sha256(args.ptxas.read_bytes()).hexdigest(),
            regalloc_opt_level=args.regalloc_opt_level,
            intervention=args.mode,
            eligible_for_fit=False,
            gpu_execution=False,
            require_archived_baseline=True,
        ),
    )
    for identity in ids:
        row = json.loads(
            (args.resource_root / "controls" / (identity + ".json")).read_text()
        )
        if row["case"] != cases[identity]:
            raise ValueError("Control identity differs from manifest")
        sources, count = variants(row, args.mode)
        folder = args.output / identity
        folder.mkdir()
        for name, ptx in sources.items():
            target = re.search(r"^\.target\s+(sm_\d+a?)\s*$", ptx, re.M)
            if not target:
                raise ValueError("Unsupported control PTX target")
            path = folder / (name + ".ptx")
            path.write_text(ptx)
            binary = path.with_suffix(".cubin")
            command = [
                str(args.ptxas),
                "-lineinfo",
                "-v",
                *(
                    []
                    if args.regalloc_opt_level is None
                    else [f"--regAllocOptLevel={args.regalloc_opt_level}"]
                ),
                "--gpu-name=" + target[1],
                str(path),
                "-o",
                str(binary),
            ]
            with path.with_suffix(".log").open("w") as log:
                subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=300,
                )
            resources = subprocess.run(
                [str(args.cuobjdump), "--dump-resource-usage", str(binary)],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout
            sass = subprocess.run(
                [str(args.cuobjdump), "--dump-sass", str(binary)],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout
            _write(
                path.with_suffix(".json"),
                dict(
                    role="control",
                    case=row["case"],
                    variant=name,
                    rewritten_fmas=count
                    if name != "original" and args.mode != "unroll"
                    else 0,
                    unrolled_trips=count if name == "unrolled" else 0,
                    command=command,
                    ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(),
                    cubin_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    archived_cubin_sha256=row["cubin_sha256"],
                    matches_archived_cubin=hashlib.sha256(
                        binary.read_bytes()
                    ).hexdigest()
                    == row["cubin_sha256"],
                    matches_archived_sass=sass == row["artifacts"].get("sass"),
                    resources=resources,
                    sass=sass,
                    eligible_for_fit=False,
                    numerical_validation="not executed",
                ),
            )
            if name == "original" and (
                hashlib.sha256(binary.read_bytes()).hexdigest() != row["cubin_sha256"]
                or sass != row["artifacts"].get("sass")
            ):
                raise RuntimeError(
                    "Original assembly does not reproduce archived control; preserve outputs and verify toolchain/options before intervention"
                )
            print(identity, name, resources.strip(), flush=True)


if __name__ == "__main__":
    main()
