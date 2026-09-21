import pytest

from triton_viz.performance.gpu_shared_dependencies import shared_dependency_barriers


def access(name, *, reads=(), writes=()):
    return dict(id=name, kind="access", reads=reads, writes=writes)


def scratch(name, region=(0, 128), scope="cta"):
    return dict(id=name, kind="scratch", scratch=region, internal_sync=scope)


@pytest.mark.parametrize(
    "first,second,expected",
    [
        ("writes", "reads", "RAW"),
        ("reads", "writes", "WAR"),
        ("writes", "writes", "WAW"),
        ("reads", "reads", None),
    ],
)
def test_overlap_requires_write_and_records_cause(first, second, expected):
    result = shared_dependency_barriers(
        [
            access("a", **{first: [(0, 64)]}),
            access("b", **{second: [(32, 96)]}),
        ]
    )
    assert result["boundary_barrier_count"] == int(expected is not None)
    if expected:
        assert result["operations"][1]["hazards"] == [
            dict(producer="a", kind=expected, overlap=[32, 64])
        ]


def test_scratch_reuse_not_constant_reduction_penalty():
    prior = access("input_dot", reads=[(0, 4096)])
    reused = shared_dependency_barriers([prior, scratch("max"), scratch("sum")])
    assert reused["boundary_barrier_count"] == 2
    assert reused["operations"][1]["hazards"][0]["kind"] == "WAR"
    disjoint = shared_dependency_barriers(
        [prior, scratch("max", (4096, 4224)), scratch("sum", (4224, 4352))]
    )
    assert disjoint["boundary_barrier_count"] == 0


def test_warp_sync_does_not_clear_unrelated_pending_cta_accesses():
    for scope, count in [("warp", 1), ("cta", 0)]:
        result = shared_dependency_barriers(
            [
                access("earlier", reads=[(256, 384)]),
                scratch("convert", scope=scope),
                access("later", writes=[(256, 384)]),
            ]
        )
        assert result["boundary_barrier_count"] == count


def test_explicit_barrier_clears_accesses_and_touching_intervals_do_not_overlap():
    result = shared_dependency_barriers(
        [
            access("a", writes=[(0, 64)]),
            access("b", reads=[(64, 128)]),
            dict(id="sync", kind="cta_barrier"),
            access("c", reads=[(0, 64)]),
        ]
    )
    assert result["boundary_barrier_count"] == 0


@pytest.mark.parametrize(
    "operation",
    [
        access("bad", reads=[(0, 0)]),
        access("bad", reads=[(-1, 1)]),
        access("bad", reads=[(0, True)]),
        dict(id="bad", kind="async_wait"),
        scratch("bad", scope="unknown"),
    ],
)
def test_unverified_effects_fail_explicitly(operation):
    with pytest.raises(ValueError):
        shared_dependency_barriers([operation])
