"""Fixed control counter matrices; capacity groups are independent CV units."""

import json

from .cases import CONFIGS


def cache_declaration(matrix="legacy"):
    if matrix == "legacy":
        return dict(
            role="control",
            working_set_mib=[3, 12, 48],
            evictions=["none", "zero", "read"],
            l2_bytes=25165824,
            block_elements=1024,
            launches=3,
            counter_sector_bytes=32,
        )
    if matrix != "capacity":
        raise ValueError("Unknown declared cache control matrix")
    declaration = json.loads((CONFIGS / "cache_capacity_control.json").read_text())
    if (
        declaration.get("role") != "control"
        or declaration.get("schema") != "triton-viz.gpu-cache-capacity-controls.v1"
    ):
        raise ValueError("Invalid cache control declaration")
    return declaration
