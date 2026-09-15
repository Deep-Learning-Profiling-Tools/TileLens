"""Equal source work, one serial chain versus two independent chains.

Every stage is stored, keeping both chains observable. These are source-level
dependency controls; GPU compilation must separately verify retained work.
"""

import triton
import triton.language as tl


@triton.jit
def _step(x, carry, MODE: tl.constexpr, BLOCK: tl.constexpr):
    value = x + carry
    if MODE == 0:
        return value * 0.5
    elif MODE == 1:
        return tl.exp(value * 0.125)
    elif MODE == 2:
        return tl.sum(value, 0) * (1.0 / BLOCK)
    else:
        return tl.max(value, 0)


@triton.jit
def paired(
    X,
    Y,
    BLOCK: tl.constexpr,
    DEPTH: tl.constexpr,
    MODE: tl.constexpr,
    SERIAL: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    if MODE < 2:
        a = tl.full((BLOCK,), 0, tl.float32)
        b = tl.full((BLOCK,), 0, tl.float32)
    else:
        a = tl.full((), 0, tl.float32)
        b = tl.full((), 0, tl.float32)
    for step in tl.static_range(DEPTH):
        x = tl.load(X + (2 * step) * BLOCK + offsets).to(tl.float32)
        y = tl.load(X + (2 * step + 1) * BLOCK + offsets).to(tl.float32)
        a = _step(x, a, MODE, BLOCK)
        b = _step(y, a if SERIAL else b, MODE, BLOCK)
        if MODE < 2:
            tl.store(Y + (2 * step) * BLOCK + offsets, a)
            tl.store(Y + (2 * step + 1) * BLOCK + offsets, b)
        else:
            tl.store(Y + 2 * step, a)
            tl.store(Y + 2 * step + 1, b)
        if SERIAL:
            a = b


OPERATIONS = ("alu", "sfu", "sum", "max")


def prepare(case, device):
    import torch
    from .kernels import _values

    block, depth = case["block"], case["repeat"]
    mode = OPERATIONS.index(case["operation"])
    x = _values((2 * depth, block), getattr(torch, case["dtype"])).to(device)
    # Accumulation/output precision stays FP32; dtype varies input storage only.
    out = torch.empty(
        (2 * depth, block) if mode < 2 else (2 * depth,),
        dtype=torch.float32,
        device=device,
    )
    return paired, (1,), (x, out, block, depth, mode, case["topology"] == "serial"), out


def check_output(case, output):
    import torch
    from .kernels import _values

    block, depth = case["block"], case["repeat"]
    x = _values((2 * depth, block), getattr(torch, case["dtype"])).float()
    a, b = torch.tensor(0.0), torch.tensor(0.0)

    def step(value, carry):
        value = value + carry
        if case["operation"] == "alu":
            return value * 0.5
        if case["operation"] == "sfu":
            return torch.exp(value * 0.125)
        if case["operation"] == "sum":
            return value.sum() / block
        return value.max()

    stages = []
    for index in range(depth):
        a = step(x[2 * index], a)
        b = step(x[2 * index + 1], a if case["topology"] == "serial" else b)
        stages.extend((a, b))
        if case["topology"] == "serial":
            a = b
    torch.testing.assert_close(output.cpu(), torch.stack(stages), rtol=1e-5, atol=1e-6)
