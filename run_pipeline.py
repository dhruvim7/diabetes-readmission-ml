#!/usr/bin/env python3
"""Command-line entry point for the readmission prediction pipeline.

Examples
--------
Full run against the real UCI dataset::

    python run_pipeline.py --config config.yaml

Fast end-to-end verification on synthetic data with the real schema::

    python tests/make_synthetic.py --n 12000 --out data/raw
    python run_pipeline.py --config config.smoke.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import run_pipeline  # noqa: E402
from src.utils import GateError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproducible 30-day readmission prediction pipeline."
    )
    parser.add_argument("--config", default="config.yaml",
                        help="Path to the analysis configuration (default: config.yaml)")
    args = parser.parse_args()

    try:
        summary = run_pipeline(args.config)
    except GateError as exc:
        print(f"\nPipeline halted at a validation gate:\n  {exc}\n", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 3

    auc = summary["best_auroc"]
    print("\n" + "=" * 72)
    print(f"Cohort            : {summary['cohort_n']:,} patients "
          f"({summary['prevalence']:.2%} 30-day readmission)")
    print(f"Best model        : {summary['best_model']}")
    print(f"Hold-out AUROC    : {auc['estimate']:.4f} "
          f"(95% CI {auc['ci_low']:.4f}-{auc['ci_high']:.4f})")
    print("=" * 72 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
