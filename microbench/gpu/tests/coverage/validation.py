"""Fresh composed validation bodies, never used as calibration controls."""

import triton
import triton.language as tl


@triton.jit
def validation(X, Y, BLOCK: tl.constexpr, MODE: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offset).to(tl.float32)
    if MODE == 0:
        centered = x - tl.sum(x, 0) / BLOCK
        result = centered * centered + x
    elif MODE == 1:
        shifted = x - tl.max(x, 0)
        weights = tl.exp(shifted)
        result = x + weights / tl.sum(weights, 0)
    elif MODE == 2:
        mean = tl.sum(x, 0) / BLOCK
        result = x / (1.0 + tl.exp(-mean))
    else:
        maximum = tl.max(x, 0)
        result = (x - maximum) * tl.rsqrt(tl.sum(x * x, 0) / BLOCK + 1.0)
    tl.store(Y + offset, result)


def prepare(case, device):
    import torch
    from .kernels import _values

    x = _values((case["programs"], case["block"]), getattr(torch, case["dtype"])).to(
        device
    )
    output = torch.empty_like(x, dtype=torch.float32)
    return (
        validation,
        (case["programs"],),
        (x, output, case["block"], case["mode"]),
        output,
    )


def check_output(case, output):
    import torch
    from .kernels import _values

    x = _values(
        (case["programs"], case["block"]), getattr(torch, case["dtype"])
    ).float()
    if case["mode"] == 0:
        expected = (x - x.mean(-1, keepdim=True)).square() + x
    elif case["mode"] == 1:
        expected = x + torch.softmax(x, dim=-1)
    elif case["mode"] == 2:
        expected = x * torch.sigmoid(x.mean(-1, keepdim=True))
    else:
        expected = (x - x.max(-1, keepdim=True).values) * torch.rsqrt(
            x.square().mean(-1, keepdim=True) + 1.0
        )
    torch.testing.assert_close(output.cpu(), expected, rtol=1e-5, atol=1e-6)
