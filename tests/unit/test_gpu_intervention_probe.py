import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from triton_viz.tools.gpu_intervention_probe import clone_with_binary, load_control
from triton_viz.tools.gpu_packing_intervention import disable_backend_unroll


def fixture():
    compiled = SimpleNamespace(
        kernel=b"original",
        asm=dict(cubin=b"original", ptx="ptx"),
        name="geometry_dot",
        module="original_module",
        function="original_function",
        _run="original_launcher",
        n_regs=40,
        n_spills=10,
        n_max_threads=128,
        metadata=object(),
        packed_metadata=object(),
        src=object(),
        hash="original_hash",
    )
    record = dict(
        role="control",
        eligible_for_fit=False,
        case=dict(kind="geometry_dot"),
        cubin_sha256=hashlib.sha256(b"variant").hexdigest(),
        archived_cubin_sha256=hashlib.sha256(b"original").hexdigest(),
    )
    return compiled, record


def test_clone_reloads_variant_without_mutating_original_handles_or_metadata():
    original, record = fixture()
    before = vars(original).copy()
    cloned = clone_with_binary(original, record, b"variant", compiler_version="3.7.0")
    assert vars(original) == before
    assert cloned.kernel == cloned.asm["cubin"] == b"variant"
    assert cloned.asm is not original.asm
    assert cloned.module is cloned.function is cloned._run is None
    assert cloned.metadata is original.metadata
    assert cloned.packed_metadata is original.packed_metadata
    assert cloned.src is original.src
    assert cloned.hash == record["cubin_sha256"]
    assert not hasattr(cloned, "n_regs") and not hasattr(cloned, "n_spills")


@pytest.mark.parametrize(
    "field,value",
    [
        ("role", "holdout"),
        ("eligible_for_fit", True),
        ("cubin_sha256", "wrong"),
        ("archived_cubin_sha256", "wrong"),
        ("case", dict(kind="other")),
    ],
)
def test_clone_rejects_unverified_provenance(field, value):
    compiled, record = fixture()
    record[field] = value
    with pytest.raises(ValueError):
        clone_with_binary(compiled, record, b"variant", compiler_version="3.7.0")


def test_clone_rejects_unverified_loader_version_and_entrypoint():
    compiled, record = fixture()
    with pytest.raises(ValueError):
        clone_with_binary(compiled, record, b"variant", compiler_version="3.5.1")
    compiled.name = "other"
    with pytest.raises(ValueError):
        clone_with_binary(compiled, record, b"variant", compiler_version="3.7.0")


def test_loader_recomputes_control_transform_and_rejects_tampering(tmp_path):
    resources, experiment = tmp_path / "resources", tmp_path / "experiment"
    (resources / "controls").mkdir(parents=True)
    (experiment / "case").mkdir(parents=True)
    case = dict(id="case", kind="geometry_dot", num_stages=2)
    ptx = ".address_size 64\n.visible .entry geometry_dot() { ret; }\n"
    _, record = fixture()
    record.update(
        case=case,
        variant="nounroll",
        ptx_sha256=hashlib.sha256(disable_backend_unroll(ptx).encode()).hexdigest(),
    )
    raw = dict(
        role="control",
        case=case,
        artifacts=dict(ptx=ptx),
        artifact_sha256=dict(ptx=hashlib.sha256(ptx.encode()).hexdigest()),
        cubin_sha256=record["archived_cubin_sha256"],
    )
    for root in (resources, experiment):
        (root / "manifest.json").write_text(
            json.dumps(dict(role="control", cases=[case], intervention="nounroll"))
        )
    (resources / "controls/case.json").write_text(json.dumps(raw))
    path = experiment / "case/nounroll"
    path.with_suffix(".json").write_text(json.dumps(record))
    path.with_suffix(".ptx").write_text(disable_backend_unroll(ptx))
    path.with_suffix(".cubin").write_bytes(b"variant")
    assert load_control(resources, experiment, "case", "nounroll") == (
        case,
        record,
        b"variant",
    )
    changed = copy.deepcopy(record)
    changed["ptx_sha256"] = hashlib.sha256(b"altered").hexdigest()
    path.with_suffix(".json").write_text(json.dumps(changed))
    path.with_suffix(".ptx").write_text("altered")
    with pytest.raises(ValueError, match="transformation"):
        load_control(resources, experiment, "case", "nounroll")
