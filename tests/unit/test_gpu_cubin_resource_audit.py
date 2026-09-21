import hashlib
import json

import pytest

from triton_viz.tools.gpu_cubin_resource_audit import (
    allocation_declaration,
    compare_driver,
    validate_control,
    join_sources,
)


def test_typed_allocation_and_driver_comparison():
    declaration = allocation_declaration(
        "Resource usage:\n Function dot:\n  REG:40 STACK:128 LOCAL:16 SHARED:0\n",
        kernel_name="dot",
    )
    assert declaration["local_bytes_per_thread"] == 144
    row = dict(registers_per_thread=40, local_bytes_per_thread=144)
    assert compare_driver(row, declaration)["driver_exact_match"]
    row["local_bytes_per_thread"] = 128
    assert not compare_driver(row, declaration)["driver_exact_match"]
    row.update(
        launch_status="out_of_resources",
        launch_error="shared",
        registers_per_thread=None,
        local_bytes_per_thread=None,
    )
    assert compare_driver(row, declaration) == dict(
        driver_comparable=False, driver_exact_match=None
    )
    row["local_bytes_per_thread"] = 0
    with pytest.raises(ValueError):
        compare_driver(row, declaration)


@pytest.mark.parametrize(
    "text",
    [
        "",
        " Function other:\n REG:40 STACK:0 LOCAL:0",
        " Function dot:\n REG:40 STACK:0",
        " Function dot:\n REG:40 REG:40 STACK:0 LOCAL:0",
        " Function dot:\n REG:40 STACK:0 LOCAL:0\n Function other:\n REG:40 STACK:0 LOCAL:0",
    ],
)
def test_ambiguous_or_incomplete_declarations_rejected(text):
    with pytest.raises(ValueError):
        allocation_declaration(text, kernel_name="dot")


def test_control_identity_and_ptx_fingerprint():
    ptx = ".target sm_121a\n"
    case = dict(id="control")
    row = dict(
        role="control",
        case=case,
        artifacts=dict(ptx=ptx),
        artifact_sha256=dict(ptx=hashlib.sha256(ptx.encode()).hexdigest()),
    )
    assert validate_control(row, case) == (ptx, "sm_121a")
    row["role"] = "holdout"
    with pytest.raises(ValueError):
        validate_control(row, case)
    row["role"] = "control"
    row["artifacts"]["ptx"] += "modified"
    with pytest.raises(ValueError, match="fingerprint"):
        validate_control(row, case)


def test_offline_join_retains_unlaunchable_rows_and_checks_all_fingerprints(tmp_path):
    source, resource, offline = (
        tmp_path / name for name in ("source", "resource", "offline")
    )
    for root in (source, resource, offline):
        root.mkdir()
        (root / "controls").mkdir()
    cases = [dict(id="loadable"), dict(id="unlaunchable")]
    manifest = dict(role="control", cases=cases, packages=dict(triton="3.7.0"))
    for root in (source, resource):
        (root / "manifest.json").write_text(json.dumps(manifest))
    (offline / "manifest.json").write_text(
        json.dumps(
            dict(
                role="control",
                cases=cases,
                source_manifest_sha256=hashlib.sha256(
                    (resource / "manifest.json").read_bytes()
                ).hexdigest(),
            )
        )
    )
    ptx, binary = ".target sm_121a\n", b"synthetic-control-cubin"
    for i, case in enumerate(cases):
        (source / "controls" / (case["id"] + ".json")).write_text(
            json.dumps(
                dict(
                    role="control",
                    case=case,
                    numerical_validation="passed",
                    compile_and_cuda_forbidden=True,
                )
            )
        )
        row = dict(
            role="control",
            case=case,
            kernel_name="dot",
            artifacts=dict(ptx=ptx),
            artifact_sha256=dict(ptx=hashlib.sha256(ptx.encode()).hexdigest()),
            cubin_sha256=hashlib.sha256(binary).hexdigest(),
            registers_per_thread=40,
            local_bytes_per_thread=128,
        )
        if i:
            row.update(
                launch_status="out_of_resources",
                launch_error="shared",
                registers_per_thread=None,
                local_bytes_per_thread=None,
            )
        (resource / "controls" / (case["id"] + ".json")).write_text(json.dumps(row))
        folder = offline / case["id"]
        folder.mkdir()
        (folder / "control.ptx").write_text(ptx)
        (folder / "control.cubin").write_bytes(binary)
        usage = " Function dot:\n REG:40 STACK:128 LOCAL:0\n"
        (folder / "result.json").write_text(
            json.dumps(
                dict(
                    role="control",
                    case=case,
                    eligible_for_fit=False,
                    cubin_sha256=row["cubin_sha256"],
                    ptx_sha256=row["artifact_sha256"]["ptx"],
                    resource_usage=usage,
                    declaration=allocation_declaration(usage, kernel_name="dot"),
                )
            )
        )
    result = join_sources(source, resource, offline)
    assert len(result["rows"]) == 2 and result["driver_compared"] == 1
    assert result["rows"][1]["launch_status"] == "out_of_resources"
    assert result["rows"][1]["local_bytes_per_thread"] == 128
    assert result["eligible_for_fit"] is False
    (offline / "unlaunchable" / "control.cubin").write_bytes(b"modified")
    with pytest.raises(ValueError, match="fingerprint"):
        join_sources(source, resource, offline)
