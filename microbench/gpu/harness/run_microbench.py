"""Collect GPU controls/holdouts with the existing audited pipeline."""

import sys

from triton_viz.tools.gpu_cost_model_pipeline import main as pipeline_main


def main(argv=None):
    return pipeline_main(["collect", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
