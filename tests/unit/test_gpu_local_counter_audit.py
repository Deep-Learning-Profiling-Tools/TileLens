import json

import pytest

from triton_viz.tools.gpu_local_counter_audit import audit, parse
from triton_viz.tools.gpu_local_counter_collect import METRICS
from triton_viz.tools.gpu_control_resources import selected_controls


def table(values=(0, 32)):
    return "\n".join(
        ['"ID","Kernel Name","Metric Name","Metric Value"']
        + [
            f'"0","geometry_dot","{metric}","{value}"'
            for metric, value in zip(METRICS, values)
        ]
    )


def test_zero_local_traffic_is_valid_and_both_counters_required():
    assert parse(table()) == {"LDL": 0, "STL": 32}
    with pytest.raises(ValueError):
        parse(table((0,)))


@pytest.mark.parametrize(
    "text",
    [
        table((float("nan"), 32)),
        table((-1, 32)),
        table((1.5, 32)),
        table().replace('"geometry_dot"', '"target"'),
        table().replace('"0","geometry_dot"', '"1","geometry_dot"'),
        table() + "\n" + table().splitlines()[-1],
    ],
)
def test_reject_invalid_or_unexpected_counters(text):
    with pytest.raises(ValueError):
        parse(text)


def test_missing_controls_retained_and_holdout_manifest_rejected(tmp_path):
    counters, resources = tmp_path / "counters", tmp_path / "resources"
    counters.mkdir()
    resources.mkdir()
    manifest = dict(
        role="control",
        cases=selected_controls("pressure"),
        hardware=dict(uuid="test-gpu"),
        metrics=list(METRICS),
    )
    for path in (counters, resources):
        (path / "manifest.json").write_text(json.dumps(manifest))
    result = audit(counters, resources)
    assert result["count"] == 32 and not result["complete"]
    assert result["compared_count"] == 0 and not result["eligible_for_fit"]
    assert all(r["counter_status"] == "incomplete" for r in result["rows"])
    manifest["role"] = "holdout"
    (resources / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="control manifests"):
        audit(counters, resources)
