import pytest

from microbench.gpu.common.cases import load_cases, paired_controls
from microbench.gpu.harness.audit_paired import audit_pair


PAIRS = paired_controls()


def test_coverage_preserves_originals_and_adds_grouped_pairs():
    cases = load_cases("coverage", "control")
    assert len(cases) == 480
    assert len(PAIRS) == 64
    assert cases[416:] == PAIRS
    assert len({c["id"] for c in cases}) == 480
    assert len(load_cases("coverage", "holdout")) == 32
    for index in range(0, len(PAIRS), 2):
        a, b = PAIRS[index : index + 2]
        assert a["cv_group"] == b["cv_group"]
        assert a["programs"] == b["programs"] == 1


@pytest.mark.parametrize(
    "index", range(0, len(PAIRS), 2), ids=[c["pair_id"] for c in PAIRS[::2]]
)
def test_cpu_paired_numerics_work_and_dependencies(index, monkeypatch):
    import triton
    import torch

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU audit must not compile or initialize CUDA")

    monkeypatch.setattr(triton.compiler, "compile", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    result = audit_pair(PAIRS[index : index + 2])
    assert result["passed"]
    assert result["serial_path"] == 2 * result["parallel_path"]
    case = PAIRS[index]
    per_stage = {
        "alu": 2,
        "sfu": 1,
        "sum": case["block"].bit_length() - 1,
        "max": case["block"].bit_length() - 1,
    }[case["operation"]]
    assert result["parallel_path"] == per_stage * case["repeat"]
    output_width = case["block"] if case["operation"] in {"alu", "sfu"} else 1
    input_bytes = 2 if case["dtype"] == "bfloat16" else 4
    assert result["bytes"] == 2 * case["repeat"] * (
        case["block"] * input_bytes + output_width * 4
    )


def test_audit_rejects_disconnected_store_values(monkeypatch):
    from triton_viz.performance import triton_observe

    original = triton_observe.observe

    def disconnected(*args, **kwargs):
        source = original(*args, **kwargs)
        for event in source["events"]:
            if event["op"] in {"store", "raw_store"}:
                event["dependencies"] = []
        return source

    monkeypatch.setattr(triton_observe, "observe", disconnected)
    with pytest.raises(ValueError, match="disconnected"):
        audit_pair(PAIRS[:2])


def test_coverage_controls_do_not_read_holdout_declarations(monkeypatch):
    from pathlib import Path

    original = Path.read_text

    def guarded(path, *args, **kwargs):
        assert "holdout" not in path.name
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    assert len(load_cases("coverage", "control")) == 480
