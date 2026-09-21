import json

import pytest

from triton_viz.tools import gpu_store_exchange_audit
from triton_viz.tools.gpu_store_exchange_audit import epilogue_exchange


def test_exchange_rounds_do_not_count_pipeline_barriers_as_output_stores():
    ptx = """
st.shared.v4.b32 [%r0], {%r1, %r2, %r3, %r4};
bar.sync 0;
fma.rn.f32x2 %rd0, %rd1, %rd2, %rd3;
bar.sync 0;
cp.async.cg.shared.global [%r0], [%rd1], 16;
bar.sync 0;
st.shared.v4.b32 [%r0], {%r1, %r2, %r3, %r4};
bar.sync 0;
ld.shared.v4.b32 {%r1, %r2, %r3, %r4}, [%r0];
bar.sync 0;
st.shared.v4.b32 [%r0], {%r1, %r2, %r3, %r4};
bar.sync 0;
ld.shared.v4.b32 {%r1, %r2, %r3, %r4}, [%r0];
"""
    assert epilogue_exchange(ptx) == dict(
        shared_store_rounds=2, shared_store_instructions=2, shared_load_instructions=2
    )
    with pytest.raises(ValueError, match="Missing IEEE"):
        epilogue_exchange("// fma.rn.f32 fake;\nbar.sync 0;")


def test_control_labels_cannot_change_prediction_and_no_cases_are_dropped(
    tmp_path, monkeypatch
):
    resources, sources = tmp_path / "resources", tmp_path / "sources"
    cases = [dict(id=name, kind="geometry_dot") for name in ("ieee", "tensor")]
    for root in (resources, sources):
        (root / "controls").mkdir(parents=True)
        (root / "manifest.json").write_text(
            json.dumps(dict(role="control", cases=cases, packages=dict(triton="3.7.0")))
        )
    for case in cases:
        (sources / "controls" / f"{case['id']}.json").write_text(
            json.dumps(
                dict(
                    role="control",
                    case=case,
                    compile_and_cuda_forbidden=True,
                    numerical_validation="passed",
                    source_precision=[
                        ["fp32", "fp32", "ieee" if case["id"] == "ieee" else "tf32"]
                    ],
                    source_features=dict(threads_per_program=128),
                    dot_shapes=[[[128, 32], [32, 256]]],
                    ood_reasons=[],
                )
            )
        )
    monkeypatch.setattr(
        gpu_store_exchange_audit,
        "resource_audit",
        lambda _: dict(complete=True, rows=[dict(case=c) for c in cases]),
    )
    path = resources / "controls" / "ieee.json"

    def write_label(rounds):
        path.write_text(
            json.dumps(
                dict(
                    artifacts=dict(
                        ptx="fma.rn.f32 %f0, %f1, %f2, %f3;\n"
                        + "st.shared.b32 [%r0], %r1;\nbar.sync 0;\n" * rounds
                    )
                )
            )
        )

    write_label(32)
    first = gpu_store_exchange_audit.audit(resources, sources)
    assert (first["count"], first["compared_count"], first["exact_count"]) == (2, 1, 1)
    assert first["rows"][1]["applicable"] is False
    write_label(1)
    second = gpu_store_exchange_audit.audit(resources, sources)
    assert second["exact_count"] == 0
    assert first["rows"][0]["prediction"] == second["rows"][0]["prediction"]
    manifest = sources / "manifest.json"
    data = json.loads(manifest.read_text())
    data["role"] = "holdout"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="control manifests"):
        gpu_store_exchange_audit.audit(resources, sources)
