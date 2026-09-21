import copy

import pytest

from triton_viz.tools.gpu_cold_control_audit import validate_batch


def row():
    samples = []
    for i in range(11):
        records = [
            dict(
                name=name,
                start_ns=i * 1000 + start,
                end_ns=i * 1000 + end,
                device=0,
                context=1,
                stream=2,
            )
            for name, start, end in [("sweep", 1, 100), ("control", 101, 201)]
        ]
        samples.append(dict(records=records, latency_us=0.1))
    return dict(
        role="control",
        case=dict(id="declared"),
        contaminated=False,
        monitoring=dict(contaminated=False, rejection_reasons=[], samples=[{}, {}]),
        dropped_records=0,
        numerical_validation="passed",
        graph_kernel_nodes=22,
        launch_mode="graph_eviction_control_pairs",
        library_sha256="declared-digest",
        metric="cupti_hes_kernel_us_eviction_unvalidated",
        eligible_for_fit=False,
        l2_capacity_bytes=24,
        eviction_bytes=48,
        samples=samples,
        median_us=0.1,
        relative_span=0,
        unstable=False,
    )


def test_recompute_every_kernel_only_interval():
    value = row()
    before = copy.deepcopy(value)
    assert (
        validate_batch(value, value["case"], library_sha256="declared-digest")[
            "latency_us"
        ]
        == 0.1
    )
    assert value == before
    value["samples"][0]["latency_us"] = 0.2
    with pytest.raises(ValueError, match="kernel-only"):
        validate_batch(value, value["case"], library_sha256="declared-digest")


@pytest.mark.parametrize(
    "field,value",
    [
        ("role", "holdout"),
        ("monitoring", None),
        ("metric", "cuda_graph_steady_cache_kernel_us"),
        ("library_sha256", "changed"),
        ("eviction_bytes", 24),
        ("dropped_records", 1),
        ("eligible_for_fit", True),
    ],
)
def test_unverified_batches_cannot_be_relabelled(field, value):
    record = row()
    record[field] = value
    with pytest.raises(ValueError):
        validate_batch(record, record["case"], library_sha256="declared-digest")


def test_chunked_protocol_cannot_be_silently_mixed():
    record = row()
    record["launch_mode"] = "chunked_graph_eviction_control_pairs"
    with pytest.raises(ValueError, match="provenance"):
        validate_batch(record, record["case"], library_sha256="declared-digest")
