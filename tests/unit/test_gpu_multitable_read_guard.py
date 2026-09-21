import json

import pytest

from triton_viz.tools.gpu_fit_guarded import audited_json_reads


@pytest.mark.parametrize("binary", [False, True])
def test_explicit_file_guard_rejects_undeclared_siblings_and_records_failure(
    tmp_path, binary
):
    control, other = tmp_path / "control.json", tmp_path / "other.json"
    control.write_text('{"role":"control"}')
    other.write_text('{"role":"holdout"}')
    report = tmp_path / "reads.json"
    with pytest.raises(AssertionError, match="undeclared"):
        with audited_json_reads([control], report):
            assert control.read_bytes() if binary else control.read_text()
            other.read_bytes() if binary else other.read_text()
    evidence = json.loads(report.read_text())
    assert evidence["reads"] == [str(control.resolve())]
    assert evidence["blocked"] == [str(other.resolve())]
    with pytest.raises(ValueError, match="fresh"):
        with audited_json_reads([control], report):
            pass
