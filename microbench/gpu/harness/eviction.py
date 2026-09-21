"""Declared persistent L2-sweep control for collector/counter diagnostics.

Touch exactly the same allocated byte range with one program per hardware SM.
This reduces grid cardinality, but cold-cache behavior still requires counters.
It is not a calibrated cache policy and is never part of the timed interval.
"""

import triton as tr
import triton.language as tl


@tr.jit
def persistent_eviction(
    buffer, size: tl.constexpr, BLOCK: tl.constexpr, PROGRAMS: tl.constexpr
):
    first = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    stride: tl.constexpr = PROGRAMS * BLOCK
    for base in range(0, (size + stride - 1) // stride):
        offsets = first + base * stride
        tl.store(buffer + offsets, 0, mask=offsets < size)
