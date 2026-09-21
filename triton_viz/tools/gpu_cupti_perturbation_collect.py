"""Run the fixed native perturbation matrix under process monitoring; no fit."""

import argparse
import hashlib
import subprocess
import time
from pathlib import Path

from microbench.gpu.harness.measure import snapshot, assert_available
from triton_viz.tools.gpu_cost_model_pipeline import _write
from triton_viz.tools.gpu_cupti_perturbation_audit import audit_log


def declared_trials(workload="fma"):
    if workload not in {"fma", "tensor", "local"}:
        raise ValueError("Unknown perturbation workload")
    trials = [
        dict(
            id=f"p{programs}_i{iterations}_t{trial}_{mode}",
            programs=programs,
            iterations=iterations,
            trial=trial,
            mode=mode,
        )
        for programs in (48, 96, 384)
        for iterations in (16, 65536)
        for trial in range(4)
        for mode in (
            ("none", "software_serial")
            if trial % 2 == 0
            else ("software_serial", "none")
        )
    ]
    if workload != "fma":
        for trial in trials:
            trial.update(id=workload + "_" + trial["id"], workload=workload)
    return trials


def monitored_process(command, log, *, allowed_graphics=(), timeout=180):
    """Allow only this exact child PID; stop our child if contamination appears.

    NVML polling does not prove exclusivity between samples. Raw logs and failed
    monitoring are retained. This never signals an unrelated process.
    """
    deadline = time.monotonic() + 10
    while True:
        before = snapshot(0)
        # NVML utilization can retain the previous *completed* trial's load.
        # Wait for a verified idle baseline, never waive the utilization check.
        if before["processes"] or any(
            p["pid"] not in allowed_graphics for p in before["graphics_processes"]
        ):
            assert_available(before, allowed_graphics=allowed_graphics)
        if float(before["utilization_pct"]) <= 5 or time.monotonic() >= deadline:
            assert_available(before, allowed_graphics=allowed_graphics)
            break
        time.sleep(0.1)
    samples, errors = [before], []
    started = time.monotonic()
    with log.open("x") as output:
        child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        try:
            while True:
                try:
                    item = snapshot(0)
                    samples.append(item)
                    if any(
                        item[key] != before[key] for key in ("uuid", "driver", "index")
                    ):
                        raise ValueError("GPU identity changed during trial")
                    assert_available(
                        item, own_pid=child.pid, allowed_graphics=allowed_graphics
                    )
                except Exception as error:
                    errors.append(str(error))
                if time.monotonic() - started > timeout:
                    errors.append("native_trial_timeout")
                if errors or child.poll() is not None:
                    break
                time.sleep(0.1)
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
            child.wait()
        try:
            after = snapshot(0)
            samples.append(after)
            assert_available(
                after, own_pid=child.pid, allowed_graphics=allowed_graphics
            )
        except Exception as error:
            errors.append(str(error))
    return dict(
        returncode=child.returncode,
        child_pid=child.pid,
        samples=samples,
        contaminated=bool(errors),
        rejection_reasons=errors,
        elapsed_seconds=time.monotonic() - started,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-idle-graphics", action="store_true")
    parser.add_argument("--workload", choices=("fma", "tensor", "local"), default="fma")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Require fresh diagnostic output root")
    binary, library = (
        args.binary.resolve(strict=True),
        args.library.resolve(strict=True),
    )
    baseline = snapshot(0)
    graphics = (
        tuple(p["pid"] for p in baseline["graphics_processes"])
        if args.allow_idle_graphics
        else ()
    )
    assert_available(baseline, allowed_graphics=graphics)
    paths = {
        "binary": binary,
        "library": library,
        "native_source": Path("microbench/gpu/harness/cupti_perturbation_native.cu"),
        "collector": Path(__file__),
        "auditor": Path(__file__).with_name("gpu_cupti_perturbation_audit.py"),
    }
    hashes = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}
    trials = declared_trials(args.workload)
    _write(
        args.output / "manifest.json",
        dict(
            role="control",
            eligible_for_fit=False,
            trials=trials,
            workload=args.workload,
            hashes=hashes,
            baseline=baseline,
            allowed_graphics=graphics,
            protocol="native_globaltimer_body_vs_software_serial_cupti_direct_11x32",
        ),
    )
    native_hardware = None
    for case in trials:
        log = args.output / (case["id"] + ".log")
        result = dict(role="control", eligible_for_fit=False, case=case)
        try:
            if any(
                hashlib.sha256(p.read_bytes()).hexdigest() != hashes[k]
                for k, p in paths.items()
            ):
                raise ValueError("Protocol artifact changed during collection")
            result["monitoring"] = monitored_process(
                [
                    str(binary),
                    case["mode"],
                    str(library),
                    str(case["programs"]),
                    str(case["iterations"]),
                ],
                log,
                allowed_graphics=graphics,
            )
            result["log_sha256"] = hashlib.sha256(log.read_bytes()).hexdigest()
            if (
                result["monitoring"]["contaminated"]
                or result["monitoring"]["returncode"] != 0
            ):
                raise ValueError("Rejected process monitoring or failed native trial")
            result["audit"] = audit_log(
                log.read_text(),
                mode=case["mode"],
                programs=case["programs"],
                iterations=case["iterations"],
                workload=args.workload,
            )
            observed_hardware = {
                k: int(result["audit"]["metadata"][k]) for k in ("l2_bytes", "sm_count")
            }
            if native_hardware is not None and native_hardware != observed_hardware:
                raise ValueError("Native hardware properties changed across trials")
            native_hardware = observed_hardware
            result["status"] = "complete"
        except Exception as error:
            result.update(status="failed", error=str(error))
            _write(args.output / (case["id"] + ".json"), result)
            raise
        _write(args.output / (case["id"] + ".json"), result)
        print(case["id"], "complete diagnostic_only", flush=True)


if __name__ == "__main__":
    main()
