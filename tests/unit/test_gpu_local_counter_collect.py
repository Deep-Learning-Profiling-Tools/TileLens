from triton_viz.tools.gpu_control_resources import selected_controls
from triton_viz.tools.gpu_local_counter_collect import METRICS, commands


def test_collects_all_declared_controls_without_timing_or_clock_changes(tmp_path):
    result = commands("ncu", tmp_path)
    assert len(result) == 32
    assert {c[c.index("--case-id") + 1] for c in result} == {
        c["id"] for c in selected_controls("pressure")
    }
    for command in result:
        for flag, value in (
            ("--clock-control", "none"),
            ("--cache-control", "all"),
            ("--replay-mode", "kernel"),
            ("--profile-from-start", "off"),
            ("--metrics", ",".join(METRICS)),
            ("--suite", "pressure"),
        ):
            assert command[command.index(flag) + 1] == value
        assert "--allow-idle-graphics" not in command
    assert all(
        "--allow-idle-graphics" in c
        for c in commands("ncu", tmp_path, allow_idle_graphics=True)
    )
