"""Matched dot controls: tile geometry, cross-program reuse and stage depth.

Every program executes the same FLOPs and loads. Reuse changes only the input
allocation/address overlap, not values, accumulator dependency or output work.
"""

import triton
import triton.language as tl


@triton.jit
def geometry_dot(
    A,
    B,
    C,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    REPEAT: tl.constexpr,
    PRECISION: tl.constexpr,
    SHARE_A: tl.constexpr,
    SHARE_B: tl.constexpr,
):
    pid = tl.program_id(0)
    m, n, k = tl.arange(0, BM), tl.arange(0, BN), tl.arange(0, BK)
    ap = 0 if SHARE_A else pid
    bp = 0 if SHARE_B else pid
    acc = tl.zeros((BM, BN), tl.float32)
    for step in range(REPEAT):
        a = tl.load(A + (ap * REPEAT + step) * BM * BK + m[:, None] * BK + k[None, :])
        b = tl.load(B + (bp * REPEAT + step) * BK * BN + k[:, None] * BN + n[None, :])
        acc = tl.dot(a, b, acc, input_precision=PRECISION)
    tl.store(C + pid * BM * BN + m[:, None] * BN + n[None, :], acc)


def operands(case):
    import torch
    from .kernels import _values

    p, r = case["programs"], case["repeat"]
    bm, bn, bk = (case[k] for k in ("bm", "bn", "bk"))
    dtype = getattr(torch, case["dtype"])
    # All programs get identical values, even in the disjoint-address control.
    # Therefore paired output/reference equality cannot depend on reuse mode.
    a = _values((1, r, bm, bk), dtype)
    b = _values((1, r, bk, bn), dtype)
    return (
        a.repeat(1 if case["reuse"] in {"a", "ab"} else p, 1, 1, 1),
        b.repeat(1 if case["reuse"] == "ab" else p, 1, 1, 1),
    )


def prepare(case, device):
    import torch

    a, b = operands(case)
    p, r = case["programs"], case["repeat"]
    bm, bn, bk = (case[k] for k in ("bm", "bn", "bk"))
    out = torch.empty((p, bm, bn), dtype=torch.float32, device=device)
    args = (
        a.to(device),
        b.to(device),
        out,
        bm,
        bn,
        bk,
        r,
        case["precision"],
        case["reuse"] in {"a", "ab"},
        case["reuse"] == "ab",
    )
    return geometry_dot, (p,), args, out


def check_output(case, output):
    import torch

    a, b = operands(case)
    expected = (a[:1].float() @ b[:1].float()).sum(1)
    expected = expected.expand(case["programs"], -1, -1)
    torch.testing.assert_close(output.cpu(), expected, rtol=0.01, atol=0.002)
