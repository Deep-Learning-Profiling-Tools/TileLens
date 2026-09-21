import json

import pytest

from triton_viz.tools import gpu_source_layout_audit


def test_compiler_labels_check_but_cannot_change_source_layout(tmp_path, monkeypatch):
    resources, sources = tmp_path / "resources", tmp_path / "sources"
    case = dict(id="control")
    for root in (resources, sources):
        (root / "controls").mkdir(parents=True)
        (root / "manifest.json").write_text(
            json.dumps(
                dict(role="control", cases=[case], packages=dict(triton="3.7.0"))
            )
        )
    source = dict(
        role="control",
        case=case,
        numerical_validation="passed",
        compile_and_cuda_forbidden=True,
        source_precision=[["fp32", "fp32", "ieee"]],
        source_features=dict(threads_per_program=128),
        dot_shapes=[[[32, 64], [64, 64]]],
    )
    path = sources / "controls" / "control.json"
    path.write_text(json.dumps(source))
    layout = dict(
        sizePerThread=[4, 4], threadsPerWarp=[2, 16], warpsPerCTA=[4, 1], order=[1, 0]
    )
    monkeypatch.setattr(
        gpu_source_layout_audit,
        "resource_audit",
        lambda _: dict(
            complete=True,
            rows=[
                dict(
                    case=case,
                    blocked_dot_fragments=[
                        dict(supported=True, shape=[32, 64, 64], layout=layout)
                    ],
                )
            ],
        ),
    )
    first = gpu_source_layout_audit.audit(resources, sources)
    assert first["exact_layout_count"] == 1
    layout["order"] = [0, 1]
    second = gpu_source_layout_audit.audit(resources, sources)
    assert second["exact_layout_count"] == 0
    assert (
        first["rows"][0]["dots"][0]["source_prediction"]
        == second["rows"][0]["dots"][0]["source_prediction"]
    )
    source["compile_and_cuda_forbidden"] = False
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="Unverified source"):
        gpu_source_layout_audit.audit(resources, sources)
