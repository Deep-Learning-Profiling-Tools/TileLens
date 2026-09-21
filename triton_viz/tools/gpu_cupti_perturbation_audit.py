"""Validate complete native perturbation logs; never admit latency fit data."""

import statistics
import argparse
import hashlib
import json
from pathlib import Path

from microbench.gpu.harness.cupti import validate_timestamps


def audit_root(root):
    from triton_viz.tools.gpu_cupti_perturbation_collect import declared_trials

    manifest = json.loads((root / "manifest.json").read_text())
    workload = manifest.get("workload", "fma")
    cases = declared_trials(workload)
    if (
        manifest.get("role") != "control"
        or manifest.get("eligible_for_fit") is not False
        or manifest["trials"] != cases
    ):
        raise ValueError("Require the full declared control-only matrix")
    rows = []
    for case in cases:
        row = json.loads((root / (case["id"] + ".json")).read_text())
        raw = (root / (case["id"] + ".log")).read_bytes()
        if (
            row.get("role") != "control"
            or row.get("eligible_for_fit") is not False
            or row["case"] != case
            or row.get("status") != "complete"
            or row["log_sha256"] != hashlib.sha256(raw).hexdigest()
        ):
            raise ValueError("Incomplete, relabeled or changed trial")
        monitor = row["monitoring"]
        if (
            monitor["contaminated"]
            or monitor["rejection_reasons"]
            or monitor["returncode"] != 0
            or len(monitor["samples"]) < 2
        ):
            raise ValueError("Rejected or incomplete monitoring")
        for sample in monitor["samples"]:
            if (
                any(
                    sample[k] != manifest["baseline"][k]
                    for k in ("uuid", "driver", "index")
                )
                or any(p["pid"] != monitor["child_pid"] for p in sample["processes"])
                or any(
                    p["pid"] not in manifest["allowed_graphics"]
                    for p in sample["graphics_processes"]
                )
            ):
                raise ValueError("Monitoring identity mismatch or foreign process")
        audited = audit_log(
            raw.decode(),
            mode=case["mode"],
            programs=case["programs"],
            iterations=case["iterations"],
            workload=workload,
        )
        if audited != row["audit"]:
            raise ValueError("Stored summary differs from raw intervals")
        rows.append(dict(case=case, audit=audited))
    paired = []
    for programs in (48, 96, 384):
        for iterations in (16, 65536):
            pairs = []
            for trial in range(4):
                modes = {
                    r["case"]["mode"]: r["audit"]
                    for r in rows
                    if (
                        r["case"]["programs"],
                        r["case"]["iterations"],
                        r["case"]["trial"],
                    )
                    == (programs, iterations, trial)
                }
                off, on = modes["none"], modes["software_serial"]
                body_off, body_on = (
                    off["body_envelope"]["median_us"],
                    on["body_envelope"]["median_us"],
                )
                cupti = on["cupti_kernel"]["median_us"]
                pairs.append(
                    dict(
                        trial=trial,
                        body_off_us=body_off,
                        body_on_us=body_on,
                        cupti_us=cupti,
                        body_change_pct=100 * (body_on / body_off - 1),
                        cupti_minus_body_on_us=cupti - body_on,
                    )
                )
            paired.append(
                dict(
                    programs=programs,
                    iterations=iterations,
                    pairs=pairs,
                    median_body_change_pct=statistics.median(
                        p["body_change_pct"] for p in pairs
                    ),
                )
            )
    return dict(
        role="control",
        eligible_for_fit=False,
        count=len(rows),
        paired=paired,
        caveat="All four pairs retained. Instrumented-body perturbation diagnostic only; no latency calibration admission.",
    )


def audit_log(text, *, mode, programs, iterations, workload="fma"):
    if workload not in {"fma", "tensor", "local"}:
        raise ValueError("Unknown perturbation workload")
    if (
        mode not in {"none", "software_serial"}
        or programs not in (48, 96, 384)
        or iterations not in (16, 65536)
    ):
        raise ValueError("Unknown declared perturbation control")
    metadata = {}
    pending, records, deliveries, bodies = [], [], [], []
    for line in text.splitlines():
        parts = line.split(",")
        if parts[0] == "cupti":
            if len(parts) != 9 or mode != "software_serial":
                raise ValueError("Unexpected CUPTI record")
            sample, index, start, end, device, context, stream = map(int, parts[1:8])
            if index != len(pending) or index not in (0, 1):
                raise ValueError("Missing or duplicate CUPTI record")
            name = "perturbation_eviction" if index == 0 else "perturbation_body"
            if name not in parts[8]:
                raise ValueError("Unexpected CUPTI kernel name")
            pending.append(
                dict(
                    sample=sample,
                    start_ns=start,
                    end_ns=end,
                    device=device,
                    context=context,
                    stream=stream,
                    name=name,
                )
            )
        elif parts[0] == "delivery":
            if len(parts) != 5 or mode != "software_serial":
                raise ValueError("Unexpected delivery record")
            sample, count, dropped, valid = map(int, parts[1:])
            if (
                (count, dropped, valid) != (2, 0, 1)
                or len(pending) != 2
                or any(r["sample"] != sample for r in pending)
            ):
                raise ValueError("Incomplete or rejected CUPTI delivery")
            deliveries.append(sample)
            records.extend(pending)
            pending = []
        elif parts[0] == "body":
            if len(parts) != 5:
                raise ValueError("Malformed CTA interval")
            sample, block, start, end = map(int, parts[1:])
            if (
                (sample, block) != divmod(len(bodies), programs)
                or start <= 0
                or end <= start
            ):
                raise ValueError("Missing, duplicate or invalid CTA interval")
            bodies.append((start, end))
        else:
            for field in parts:
                if "=" not in field:
                    raise ValueError("Unknown native log line")
                key, value = field.split("=", 1)
                # Both provenance and summary deliberately state ineligibility.
                if key in metadata and not (
                    key == "eligible_for_fit" and value == metadata[key] == "false"
                ):
                    raise ValueError("Duplicate native metadata")
                metadata[key] = value
    expected = dict(
        role="control",
        eligible_for_fit="false",
        mode=mode,
        programs=str(programs),
        iterations=str(iterations),
        completed="352",
        valid="1",
        close_status="0",
    )
    if workload != "fma" or "workload" in metadata:
        expected["workload"] = workload
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError("Unverified identity, numerical result or completion")
    allowed = set(expected) | {
        "l2_bytes",
        "sm_count",
        "eviction_bytes",
        "warmup_launches",
    }
    if set(metadata) != allowed:
        raise ValueError("Missing or unknown protocol metadata")
    l2, sms, eviction, warmups = (
        int(metadata[k])
        for k in ("l2_bytes", "sm_count", "eviction_bytes", "warmup_launches")
    )
    if l2 <= 0 or sms <= 0 or eviction != 2 * l2 or warmups < 1:
        raise ValueError("Invalid hardware or warmup/sweep provenance")
    if pending or len(bodies) != 352 * programs:
        raise ValueError("Incomplete measurement matrix")
    cupti = []
    if mode == "software_serial":
        if deliveries != [-1] * warmups + list(range(352)):
            raise ValueError("Missing, reordered or extra CUPTI samples")
        ordered = validate_timestamps(
            records,
            expected_names=["perturbation_eviction", "perturbation_body"]
            * (warmups + 352),
            device=0,
        )
        if records != ordered:
            raise ValueError("CUPTI records not in execution order")
        cupti = [
            (r["end_ns"] - r["start_ns"]) / 1000 for r in records[2 * warmups + 1 :: 2]
        ]
    elif records or deliveries:
        raise ValueError("Disabled mode contains CUPTI activity")
    envelopes = []
    previous_end = 0
    for offset in range(0, len(bodies), programs):
        group = bodies[offset : offset + programs]
        start, end = min(r[0] for r in group), max(r[1] for r in group)
        if start < previous_end:
            raise ValueError("Body sample envelopes overlap or reorder")
        previous_end = end
        envelopes.append((end - start) / 1000)

    def summarize(values):
        groups = [statistics.mean(values[i : i + 32]) for i in range(0, 352, 32)]
        median = statistics.median(groups)
        return dict(
            samples_us=values,
            group_means_us=groups,
            median_us=median,
            relative_span=(max(groups) - min(groups)) / median,
        )

    return dict(
        role="control",
        eligible_for_fit=False,
        metadata=metadata,
        body_envelope=summarize(envelopes),
        cupti_kernel=summarize(cupti) if cupti else None,
        # Durations only: these clocks' absolute epochs are not assumed equal.
        cupti_minus_body_us=[a - b for a, b in zip(cupti, envelopes)],
        caveat="Instrumented body envelopes omit kernel prelude/epilogue; no timing admission or uninstrumented ground truth.",
    )


def main(argv=None):
    from triton_viz.tools.gpu_cost_model_pipeline import _write

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Require fresh perturbation audit output")
    result = audit_root(args.root)
    _write(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
