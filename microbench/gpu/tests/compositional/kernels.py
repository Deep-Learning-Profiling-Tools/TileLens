"""Compositional GPU kernels with separately declared control/holdout cases.

Controls extend the pilot with dependent reductions of different depths.
Validation kernels/sizes are separate from the original pilot holdout.
"""

import triton
import triton.language as tl

from microbench.gpu.tests.primitive import kernels as pilot


@triton.jit
def reduction_chain(
    X, Y, N: tl.constexpr, BLOCK: tl.constexpr, REPEAT: tl.constexpr, MODE: tl.constexpr
):
    col = tl.arange(0, BLOCK)
    offset = tl.program_id(0) * BLOCK + col
    x = tl.load(X + offset)
    for _ in range(REPEAT):
        if MODE == 0:
            value = tl.sum(x, 0) / BLOCK
        else:
            value = tl.max(x, 0)
        x = x * 0.9 + value * 0.01
    tl.store(Y + offset, x)


@triton.jit
def validation_kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr, MODE: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offset)
    if MODE == 0:
        mean = tl.sum(x, 0) / BLOCK
        centered = x - mean
        variance = tl.sum(centered * centered, 0) / BLOCK
        result = centered * tl.rsqrt(variance + 1e-5)
    elif MODE == 1:
        shifted = x - tl.max(x, 0)
        result = shifted - tl.log(tl.sum(tl.exp(shifted), 0))
    elif MODE == 2:
        result = x + x / (1 + tl.exp(-x))
    else:
        # Two independent reductions join into a broadcast value.
        mean = tl.sum(x, 0) / BLOCK
        maximum = tl.max(x, 0)
        result = (x - mean) / (1 + maximum * maximum)
    tl.store(Y + offset, result)


def cases(role):
    from microbench.gpu.common.cases import load_cases

    return load_cases("compositional", role)


def prepare(case, device):
    import torch

    if case["kind"] not in {"reduction_chain", "validation"}:
        return pilot.prepare(case, device)
    n, block = case["n"], case["block"]
    # Nonconstant inputs make normalization/output checks meaningful.
    values = (torch.arange(n, dtype=torch.int64) % 17).float() / 50 + 0.01
    x = values.to(device)
    out = torch.empty_like(x)
    if case["kind"] == "reduction_chain":
        return (
            reduction_chain,
            (n // block,),
            (x, out, n, block, case["repeat"], case["mode"]),
            out,
        )
    return validation_kernel, (n // block,), (x, out, n, block, case["mode"]), out


def check_output(case, output):
    import torch

    if case["kind"] not in {"reduction_chain", "validation"}:
        return pilot.check_output(case, output)
    n, block, mode = case["n"], case["block"], case["mode"]
    x = ((torch.arange(n, dtype=torch.int64) % 17).float() / 50 + 0.01).reshape(
        -1, block
    )
    if case["kind"] == "reduction_chain":
        for _ in range(case["repeat"]):
            value = (
                x.mean(dim=1, keepdim=True)
                if mode == 0
                else x.max(dim=1, keepdim=True).values
            )
            x = x * 0.9 + value * 0.01
        expected = x
    elif mode == 0:
        centered = x - x.mean(dim=1, keepdim=True)
        expected = centered * torch.rsqrt(
            (centered * centered).mean(dim=1, keepdim=True) + 1e-5
        )
    elif mode == 1:
        expected = torch.log_softmax(x, dim=1)
    elif mode == 2:
        expected = x + x * torch.sigmoid(x)
    else:
        expected = (x - x.mean(dim=1, keepdim=True)) / (
            1 + x.max(dim=1, keepdim=True).values.square()
        )
    torch.testing.assert_close(
        output.cpu().reshape_as(expected), expected, rtol=2e-3, atol=2e-4
    )
