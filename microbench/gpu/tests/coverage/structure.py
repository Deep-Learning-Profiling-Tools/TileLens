"""Control-only streaming-layout and dot/normalization/dot compositions."""

import triton
import triton.language as tl


@triton.jit
def stream_dot(
    A,
    B,
    C,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    R: tl.constexpr,
    PN: tl.constexpr,
    PACKED: tl.constexpr,
    PRECISION: tl.constexpr,
):
    mi, ni = tl.program_id(0), tl.program_id(1)
    m, n, k = tl.arange(0, BM), tl.arange(0, BN), tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for step in range(R):
        if PACKED:
            ap = (mi * R + step) * BM * BK + m[:, None] * BK + k[None, :]
        else:
            ap = mi * BM * R * BK + m[:, None] * R * BK + step * BK + k[None, :]
        bp = (ni * R + step) * BK * BN + k[:, None] * BN + n[None, :]
        acc = tl.dot(tl.load(A + ap), tl.load(B + bp), acc, input_precision=PRECISION)
    tl.store(C + (mi * PN + ni) * BM * BN + m[:, None] * BN + n[None, :], acc)


@triton.jit
def composed_dot(
    A,
    B,
    V,
    C,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    MODE: tl.constexpr,
    PRECISION: tl.constexpr,
):
    pid = tl.program_id(0)
    m, n, k = tl.arange(0, BM), tl.arange(0, BN), tl.arange(0, BK)
    a = tl.load(A + pid * BM * BK + m[:, None] * BK + k[None, :])
    b = tl.load(B + pid * BK * BN + k[:, None] * BN + n[None, :])
    x = tl.dot(a, b, input_precision=PRECISION) * (BK**-0.5)
    if MODE >= 1:
        x = tl.exp(x - tl.max(x, 1)[:, None])
        x = x / tl.sum(x, 1)[:, None]
    if MODE == 2:
        v = tl.load(V + pid * BN * BN + n[:, None] * BN + n[None, :])
        x = tl.dot(x.to(a.dtype), v, input_precision=PRECISION)
    tl.store(C + pid * BM * BN + m[:, None] * BN + n[None, :], x)


def inputs(case):
    import torch
    from .kernels import _values

    p, r = case["programs"], case["repeat"]
    bm, bn, bk = (case[k] for k in ("bm", "bn", "bk"))
    dtype = getattr(torch, case["dtype"])
    if case["kind"] == "structure_stream":
        return _values((2, bm, r * bk), dtype), _values((p // 2, r * bk, bn), dtype)
    return (
        _values((p, bm, bk), dtype),
        _values((p, bk, bn), dtype),
        _values((p, bn, bn), dtype),
    )


def prepare(case, device):
    import torch

    p, r = case["programs"], case["repeat"]
    bm, bn, bk = (case[k] for k in ("bm", "bn", "bk"))
    arrays = inputs(case)
    if case["kind"] == "structure_stream":
        a, b = arrays
        packed = case["variant"] == "packed"
        if packed:
            a = a.reshape(2, bm, r, bk).permute(0, 2, 1, 3).contiguous()
        out = torch.empty((2, p // 2, bm, bn), dtype=torch.float32, device=device)
        return (
            stream_dot,
            (2, p // 2),
            (
                a.to(device),
                b.to(device),
                out,
                bm,
                bn,
                bk,
                r,
                p // 2,
                packed,
                case["precision"],
            ),
            out,
        )
    out = torch.empty((p, bm, bn), dtype=torch.float32, device=device)
    return (
        composed_dot,
        (p,),
        (
            *(a.to(device) for a in arrays),
            out,
            bm,
            bn,
            bk,
            case["variant"],
            case["precision"],
        ),
        out,
    )


def check_output(case, output):
    import torch

    arrays = inputs(case)
    a, b = arrays[:2]
    if case["kind"] == "structure_stream":
        expected = a[:, None].float() @ b[None, :].float()
    else:
        expected = (a.float() @ b.float()) * case["bk"] ** -0.5
        if case["variant"] >= 1:
            expected = expected.softmax(-1)
        if case["variant"] == 2:
            expected = expected.to(a.dtype).float() @ arrays[2].float()
    torch.testing.assert_close(output.cpu(), expected, rtol=0.01, atol=0.002)
