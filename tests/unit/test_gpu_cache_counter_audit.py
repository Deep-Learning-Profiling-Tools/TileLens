import csv
import io
import json

import pytest

from triton_viz.tools.gpu_cache_counter_audit import (
    METRICS,
    audit,
    audit_ranges,
    parse_counters,
)
from triton_viz.tools.gpu_control_resources import selected_controls
from microbench.gpu.common.cache_controls import cache_declaration


def table(hits=(1, 99, 1)):
    stream = io.StringIO()
    writer = csv.writer(stream, quoting=csv.QUOTE_ALL)
    writer.writerow(["ID", "Kernel Name", "Metric Name", "Metric Value"])
    for launch, hit in enumerate(hits):
        values = {"reads": 100, "hits": hit, "misses": 100 - hit}
        for metric, name in METRICS.items():
            writer.writerow([launch, "cache_read_control", metric, values[name]])
    return stream.getvalue()


def test_counter_parser_keeps_all_launches():
    rows = parse_counters("==PROF== preamble\n" + table())
    assert [row["hit_fraction"] for row in rows] == [0.01, 0.99, 0.01]
    assert all(row["replay_count_disagreement"] == 0 for row in rows)


@pytest.mark.parametrize(
    "text",
    [
        "incomplete",
        table().replace("cache_read_control", "target"),
        table() + table().splitlines()[-1],
        table().replace('"100"', '"nan"'),
    ],
)
def test_bad_counter_reports_fail(text):
    with pytest.raises(ValueError):
        parse_counters(text)


def test_counter_audit_requires_every_declared_case_and_keeps_attempts(tmp_path):
    assert not audit(tmp_path)["complete"]
    for mib in (3, 12, 48):
        for eviction in ("none", "zero", "read"):
            path = tmp_path / f"matrix_{mib}_{eviction}.csv"
            path.write_text(table((1, 99, 99 if eviction == "none" else 1)))
            path.with_suffix(".log").write_text("numerical=passed")
    report = audit(tmp_path)
    assert report["complete"]
    assert report["validated_eviction"] == {"zero": True, "read": True}
    assert not report["eligible_for_latency_fit"]
    (tmp_path / "matrix_3_read.csv").write_text("profiler timeout")
    incomplete = audit(tmp_path)
    assert not incomplete["complete"]
    assert incomplete["cold_to_warm_positive_control"]
    assert not any(incomplete["validated_eviction"].values())
    retry = tmp_path / "matrix_3_read_retry.csv"
    retry.write_text(table())
    retry.with_suffix(".log").write_text("numerical=passed")
    report = audit(tmp_path)
    row = next(
        r
        for r in report["rows"]
        if r["working_set_mib"] == 3 and r["eviction"] == "read"
    )
    assert len(row["attempts"]) == 2
    assert row["attempts"][0]["error"]
    assert report["complete"]
    assert len(report["rows"]) == 9


def test_control_resource_selector_cannot_request_holdout():
    assert len(selected_controls("geometry")) == 144
    assert len(selected_controls("structure")) == 96
    assert len(selected_controls("pressure")) == 32
    with pytest.raises(ValueError):
        selected_controls("holdout")


def test_range_parser_does_not_treat_aggregate_as_three_kernels(tmp_path):
    lines = table().replace("cache_read_control", "range").splitlines()
    text = "\n".join(lines[:4])
    assert len(parse_counters(text, range_mode=True)) == 1
    with pytest.raises(ValueError):
        parse_counters(text)
    assert not audit_ranges(tmp_path)["complete"]
    for mib in (3, 12, 48):
        for eviction in ("none", "zero", "read"):
            path = tmp_path / f"{mib}_{eviction}.csv"
            path.write_text(text)
            path.with_suffix(".log").write_text("numerical=passed")
    report = audit_ranges(tmp_path)
    assert report["complete"] and report["replay_counts_consistent"]
    assert report["zero_sweep_added_miss_footprints"] == 0
    assert not report["eligible_for_latency_fit"]
    assert len(report["rows"]) == 9


def test_capacity_audit_binds_hardware_and_exact_probe_geometry(tmp_path):
    declaration = cache_declaration("capacity")
    manifest = dict(
        role="control",
        matrix="capacity",
        declaration=declaration,
        hardware=dict(uuid="control-gpu", driver="test-driver", index=0),
    )
    text = "\n".join(table().replace("cache_read_control", "range").splitlines()[:4])
    for mib in declaration["working_set_mib"]:
        for eviction in declaration["evictions"]:
            path = tmp_path / f"{mib}_{eviction}.csv"
            path.write_text(text)
            path.with_suffix(".log").write_text(
                f"control cache_read: {mib} MiB, eviction={eviction}, "
                f"block={declaration['block_elements']}, "
                f"L2={declaration['l2_bytes']}, numerical=passed"
            )
    assert not audit_ranges(tmp_path, "capacity")["complete"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    result = audit_ranges(tmp_path, "capacity")
    assert result["complete"] and len(result["rows"]) == 27
    assert result["hardware"] == manifest["hardware"]
    (tmp_path / "3_zero.log").write_text("numerical=passed")
    result = audit_ranges(tmp_path, "capacity")
    assert not result["complete"] and len(result["rows"]) == 27
    assert not result["replay_counts_consistent"]
