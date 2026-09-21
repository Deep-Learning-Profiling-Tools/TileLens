"""Control-only dtype, launch, loop-depth and dot-precision experiments."""

import triton
import triton.language as tl


@triton.jit
def vector(X, Y, BLOCK: tl.constexpr, REPEAT: tl.constexpr, MODE: tl.constexpr):
    pid = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    if MODE < 3:
        x = tl.load(X + pid * BLOCK + col).to(tl.float32)
        for _ in range(REPEAT):
            if MODE == 1:
                x = x * 0.75 + 0.025
            elif MODE == 2:
                x = tl.exp(x * 0.01)
        tl.store(Y + pid * BLOCK + col, x)
    else:
        carry = 0.0
        for step in range(REPEAT):
            x = tl.load(X + (pid * REPEAT + step) * BLOCK + col).to(tl.float32)
            if MODE == 3:
                value = tl.sum(x, 0)
            else:
                value = tl.max(x, 0)
            carry = carry * 0.5 + value
        tl.store(Y + pid, carry)


@triton.jit
def dot_control(
    A,
    B,
    C,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    REPEAT: tl.constexpr,
    PRECISION: tl.constexpr,
):
    pid = tl.program_id(0)
    m, n, k = tl.arange(0, BM), tl.arange(0, BN), tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for step in range(REPEAT):
        a = tl.load(A + (pid * REPEAT + step) * BM * BK + m[:, None] * BK + k[None, :])
        b = tl.load(B + (pid * REPEAT + step) * BK * BN + k[:, None] * BN + n[None, :])
        acc = tl.dot(a, b, acc, input_precision=PRECISION)
    tl.store(C + pid * BM * BN + m[:, None] * BN + n[None, :], acc)


def cases(role):
    from microbench.gpu.common.cases import load_cases

    return load_cases("coverage", role)


def _values(shape, dtype):
    import math
    import torch

    # Binary fractions: exactly representable in BF16 and TF32, so correctness
    # validation does not need an error tolerance chosen from target results.
    x = ((torch.arange(math.prod(shape)) % 13).float() - 6) / 32
    return x.reshape(shape).to(dtype)


def prepare(case, device):
    if case["kind"].startswith("structure_"):
        from .structure import prepare as prepare_structure

        return prepare_structure(case, device)
    if case["kind"] == "geometry_dot":
        from .geometry import prepare as prepare_geometry

        return prepare_geometry(case, device)
    if case["kind"] == "coverage_validation":
        from .validation import prepare as prepare_validation

        return prepare_validation(case, device)
    if case["kind"] == "coverage_paired":
        from .paired import prepare as prepare_paired

        return prepare_paired(case, device)
    import torch

    p, repeat = case["programs"], case["repeat"]
    dtype = getattr(torch, case["dtype"])
    if case["kind"] == "coverage_dot":
        bm, bn, bk = (case[k] for k in ("bm", "bn", "bk"))
        a = _values((p, repeat, bm, bk), dtype).to(device)
        b = _values((p, repeat, bk, bn), dtype).to(device)
        out = torch.empty((p, bm, bn), dtype=torch.float32, device=device)
        return (
            dot_control,
            (p,),
            (a, b, out, bm, bn, bk, repeat, case["precision"]),
            out,
        )
    block, mode = case["block"], case["mode"]
    x = _values((p, repeat if mode >= 3 else 1, block), dtype).to(device)
    out = torch.empty((p,) if mode >= 3 else (p, block), dtype=dtype, device=device)
    return vector, (p,), (x, out, block, repeat, mode), out


def check_output(case, output):
    if case["kind"].startswith("structure_"):
        from .structure import check_output as check_structure

        return check_structure(case, output)
    if case["kind"] == "geometry_dot":
        from .geometry import check_output as check_geometry

        return check_geometry(case, output)
    if case["kind"] == "coverage_validation":
        from .validation import check_output as check_validation

        return check_validation(case, output)
    if case["kind"] == "coverage_paired":
        from .paired import check_output as check_paired

        return check_paired(case, output)
    import torch

    p, repeat = case["programs"], case["repeat"]
    dtype = getattr(torch, case["dtype"])
    if case["kind"] == "coverage_dot":
        bm, bn, bk = (case[k] for k in ("bm", "bn", "bk"))
        a = _values((p, repeat, bm, bk), dtype).float()
        b = _values((p, repeat, bk, bn), dtype).float()
        expected = (a @ b).sum(1)
    else:
        block, mode = case["block"], case["mode"]
        x = _values((p, repeat if mode >= 3 else 1, block), dtype).float()
        if mode >= 3:
            expected = torch.zeros(p)
            for step in range(repeat):
                value = x[:, step].sum(-1) if mode == 3 else x[:, step].max(-1).values
                expected = expected * 0.5 + value
        else:
            expected = x[:, 0]
            for _ in range(repeat):
                if mode == 1:
                    expected = expected * 0.75 + 0.025
                elif mode == 2:
                    expected = torch.exp(expected * 0.01)
        expected = expected.to(dtype)
    torch.testing.assert_close(output.cpu(), expected, rtol=0.01, atol=0.002)
