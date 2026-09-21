import copy

import pytest

from microbench.gpu.common.cases import load_cases
from triton_viz.tools.gpu_composition_lowering_audit import audit


def reports():
    def row(case):
        return dict(
            case=case,
            status="complete",
            registers_per_thread=40,
            local_bytes_per_thread=0,
            shared_bytes=4096,
            latency=object(),
        )  # The audit must not inspect timings.

    parents = [row(c) for c in load_cases("resource_transfer", "control")]
    components = [row(c) for c in load_cases("composition_component", "control")]
    for r in parents:
        if r["case"].get("variant") == 2:
            r["local_bytes_per_thread"] = 64
    return (
        dict(role="control", complete=True, rows=parents),
        dict(role="control", complete=True, rows=components),
    )


def test_every_control_accounted_and_no_latency_consumed():
    parents, components = reports()
    result = audit(parents, components)
    assert result["input_control_count"] == 224
    assert result["paired_control_count"] == 128
    assert len(result["unpaired_parent_ids"]) == 96
    assert len(result["rows"]) == 32
    assert all(r["local_allocation_only_in_composition"] for r in result["rows"])
    assert not result["eligible_for_fit"]
    before = copy.deepcopy(result)
    for row in parents["rows"]:
        row["latency"] = 1e30
    assert audit(parents, components) == before


@pytest.mark.parametrize(
    "mutation",
    ["holdout", "missing", "group", "geometry", "duplicate", "phase_geometry"],
)
def test_unmatched_or_untrusted_controls_rejected(mutation):
    parents, components = reports()
    if mutation == "holdout":
        components["role"] = "holdout"
    elif mutation == "missing":
        components["rows"].pop()
    elif mutation == "group":
        components["rows"][0]["case"]["cv_group"] = "leaked"
    elif mutation == "geometry":
        components["rows"][0]["case"]["bk"] = 32
    elif mutation == "duplicate":
        components["rows"].append(components["rows"][0])
    else:
        next(r for r in parents["rows"] if r["case"].get("variant") == 1)["case"][
            "bk"
        ] = 128
    with pytest.raises(ValueError):
        audit(parents, components)
