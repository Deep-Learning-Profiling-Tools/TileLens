"""Address-free working-set summaries from pre-compile source observation."""

import math

MEMORY_FEATURES = (
    "load_unique_sectors",
    "load_repeat_sectors",
    "store_sector_requests",
)
PRESSURE_FEATURE = "load_footprint_program_pressure"


def memory_features(source):
    summary = source.get("memory_working_set", {})
    empty = dict.fromkeys((*MEMORY_FEATURES, PRESSURE_FEATURE), 0.0)
    if summary.get("schema") != "triton-viz.gpu-memory-working-set.v1":
        return empty, ["missing_memory_working_set_metadata"]
    try:
        unique, requests, stores, local = (
            float(summary[k])
            for k in (
                "load_unique_sectors",
                "load_sector_requests",
                "store_sector_requests",
                "program_load_sectors_p90",
            )
        )
        if not all(
            math.isfinite(v) and v >= 0 for v in (unique, requests, stores, local)
        ):
            raise ValueError("Invalid memory count")
        if unique > requests or local > requests:
            raise ValueError("Inconsistent memory count")
        loads = sum(
            e["sectors"] for e in source["events"] if e["op"] in {"load", "raw_load"}
        )
        writes = sum(
            e["sectors"] for e in source["events"] if e["op"] in {"store", "raw_store"}
        )
        if (requests, stores) != (loads, writes):
            raise ValueError("Memory summary differs from events")
    except (KeyError, TypeError, ValueError):
        return empty, ["invalid_memory_working_set_metadata"]
    return {
        "load_unique_sectors": unique,
        "load_repeat_sectors": requests - unique,
        "store_sector_requests": stores,
        # Source-only interaction, not a claimed cache size/threshold. Its
        # coefficient and inclusion are determined exclusively by control CV.
        PRESSURE_FEATURE: unique * local,
    }, []
