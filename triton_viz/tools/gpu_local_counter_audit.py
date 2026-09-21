"""Compare all declared local-counter controls to explicit loop hypotheses."""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

from triton_viz.tools.gpu_control_resources import selected_controls
from triton_viz.tools.gpu_cost_model_pipeline import _write
from triton_viz.tools.gpu_local_counter_collect import METRICS
from triton_viz.tools.gpu_local_traffic_audit import account


def validate_monitor(path, manifest):
    record = json.loads(path.with_suffix(".monitor.json").read_text())
    if (
        record["case_id"] != path.stem
        or record["csv_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
        or record["log_sha256"]
        != hashlib.sha256(path.with_suffix(".log").read_bytes()).hexdigest()
    ):
        raise ValueError("Counter monitoring evidence hash or identity mismatch")
    monitor = record["monitoring"]
    if (
        monitor["returncode"] != 0
        or monitor["contaminated"]
        or monitor["rejection_reasons"]
        or monitor.get("own_process_group") is not True
        or len(monitor["samples"]) < 2
    ):
        raise ValueError("Rejected or incomplete process monitoring")
    for sample in monitor["samples"]:
        groups = sample.get("observed_process_groups", {})
        if (
            any(
                sample[k] != manifest["hardware"][k]
                for k in ("uuid", "driver", "index")
            )
            or any(
                p["pid"] != monitor["child_pid"]
                and groups.get(str(p["pid"])) != monitor["child_pid"]
                for p in sample["processes"]
            )
            or any(
                p["pid"] not in manifest["allowed_graphics"]
                for p in sample["graphics_processes"]
            )
        ):
            raise ValueError("Foreign process or hardware identity mismatch")


def parse(text):
    lines = text.splitlines()
    header = next((i for i, line in enumerate(lines) if line.startswith('"ID",')), None)
    if header is None:
        raise ValueError("Missing counter CSV")
    values = {}
    for row in csv.DictReader(io.StringIO("\n".join(lines[header:]))):
        metric = row["Metric Name"]
        if (
            row["ID"] != "0"
            or row["Kernel Name"] != "geometry_dot"
            or metric not in METRICS
            or metric in values
        ):
            raise ValueError("Unexpected kernel, launch or metric")
        value = float(row["Metric Value"].replace(",", ""))
        if not math.isfinite(value) or value < 0 or not value.is_integer():
            raise ValueError("Invalid sector count")
        values[metric] = int(value)
    if set(values) != set(METRICS):
        raise ValueError("Require both local counters")
    return dict(zip(("LDL", "STL"), (values[m] for m in METRICS)))


def audit(counter_root, resource_root, source_root=None, *, suite="pressure"):
    if suite not in {"pressure", "pressure_pipeline", "resource_dot"}:
        raise ValueError("Require a declared pressure control suite")
    cases = selected_controls(suite)
    manifests = [
        json.loads((p / "manifest.json").read_text())
        for p in (counter_root, resource_root)
    ]
    for index, manifest in enumerate(manifests):
        expected = (
            selected_controls("resource_transfer")
            if index == 1 and suite == "resource_dot"
            else cases
        )
        if manifest.get("role") != "control" or manifest.get("cases") != expected:
            raise ValueError("Require identical complete declared control manifests")
    if manifests[0].get("metrics") != list(METRICS) or not manifests[0].get(
        "hardware", {}
    ).get("uuid"):
        raise ValueError("Missing counter protocol identity")
    hypotheses = None
    if source_root is not None:
        from triton_viz.tools.gpu_dot_lowering_audit import audit as dot_audit

        hypotheses = {
            r["case"]["id"]: r["dot_work_loop_hypothesis"]
            for r in dot_audit(resource_root, source_root)["rows"]
            if r["applicable"]
        }
    rows = []
    for case in cases:
        row = dict(case=case)
        path = counter_root / (case["id"] + ".csv")
        try:
            if manifests[0].get("monitored"):
                validate_monitor(path, manifests[0])
            text = path.read_text()
            if (
                f"control={case['id']} numerical=passed profiler_timing_not_for_fit"
                not in path.with_suffix(".log").read_text()
            ):
                raise ValueError("Missing numerical control identity")
            row.update(
                counters=parse(text),
                counter_sha256=hashlib.sha256(text.encode()).hexdigest(),
                counter_status="complete",
            )
        except (OSError, ValueError) as exc:
            row.update(counter_status="incomplete", counter_error=str(exc))
        try:
            compiled = json.loads(
                (resource_root / "controls" / (case["id"] + ".json")).read_text()
            )
            if compiled.get("case") != case:
                raise ValueError("Compiler control identity mismatch")
            trips = case["repeat"]
            if hypotheses is not None:
                hypothesis = hypotheses[case["id"]]
                row["dot_work_loop_hypothesis"] = hypothesis
                if not hypothesis["supported"]:
                    raise ValueError("Source/SASS dot-work hypothesis unsupported")
                trips = hypothesis["loop_trips"] or 1
            row["conditional_accounting"] = account(compiled, loop_trips=trips)
            if row["counter_status"] == "complete":
                row["conditional_exact_match"] = (
                    row["counters"]
                    == row["conditional_accounting"][
                        "conditional_payload_sector_equivalents"
                    ]
                )
        except (OSError, ValueError) as exc:
            row["accounting_unsupported"] = str(exc)
        rows.append(row)
    return dict(
        role="control",
        count=len(rows),
        rows=rows,
        complete=all(r["counter_status"] == "complete" for r in rows),
        compared_count=sum("conditional_exact_match" in r for r in rows),
        conditional_exact_count=sum(
            r.get("conditional_exact_match", False) for r in rows
        ),
        eligible_for_fit=False,
        trip_hypothesis="source_dot_work_reconciled_with_control_sass"
        if hypotheses is not None
        else "declared_source_repeat",
        caveat="All controls retained. Trip hypotheses do not read counters. Dot-work consistency does not prove branch execution or active lanes. Counter agreement is a diagnostic, not a source mapping CV gate or latency release.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("pressure", "pressure_pipeline", "resource_dot"),
        default="pressure",
    )
    parser.add_argument("--counter-root", type=Path, required=True)
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Use a fresh audit output")
    result = audit(
        args.counter_root, args.resource_root, args.source_root, suite=args.suite
    )
    _write(args.output, result)
    print(
        {
            k: result[k]
            for k in ("count", "complete", "compared_count", "conditional_exact_count")
        }
    )


if __name__ == "__main__":
    main()
