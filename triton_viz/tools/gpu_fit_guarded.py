"""Fit preregistered memory candidates with a control-only JSON read audit."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from triton_viz.tools.gpu_distribution_experiments import fit


def guarded_fit(root, output, *, wave_dot=False):
    original = Path.read_text
    accessed = []
    allowed_manifest = (root / "manifest.json").resolve()
    controls = (root / "controls").resolve()

    def guarded(path, *args, **kwargs):
        if path.suffix == ".json":
            resolved = path.resolve()
            if resolved != allowed_manifest and resolved.parent != controls:
                raise AssertionError(f"Fit attempted non-control JSON read: {path}")
            accessed.append(str(resolved))
        return original(path, *args, **kwargs)

    # Keep an audit even if the CV gate rejects all candidates.
    try:
        with patch.object(Path, "read_text", guarded):
            fit(root, output, memory_working_set=True, wave_dot=wave_dot)
    finally:
        output.mkdir(parents=True, exist_ok=True)
        (output / "fit_reads.json").write_text(json.dumps(accessed, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wave-dot", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh fit output to avoid stale frozen artifacts")
    guarded_fit(args.root, args.output, wave_dot=args.wave_dot)


if __name__ == "__main__":
    main()
