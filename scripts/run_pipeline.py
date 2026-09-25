"""
Example:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --data-dir path/to/data --output-dir path/to/out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shsc import Paths, run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SHSC modelling pipeline.")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Directory containing demo_train.csv, demo_internal_val.csv, "
                             "demo_external_test.csv, and demo_clinical.csv.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for outputs (default: <root>/outputs).")
    parser.add_argument("--train", type=Path, default=None, dest="train_csv")
    parser.add_argument("--internal", type=Path, default=None, dest="internal_csv")
    parser.add_argument("--external", type=Path, default=None, dest="external_csv")
    parser.add_argument("--clinical", type=Path, default=None, dest="clinical_csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = Paths(output_dir=args.output_dir)

    if args.data_dir is not None:
        paths.train_csv = args.data_dir / "demo_train.csv"
        paths.internal_csv = args.data_dir / "demo_internal_val.csv"
        paths.external_csv = args.data_dir / "demo_external_test.csv"
        paths.clinical_csv = args.data_dir / "demo_clinical.csv"
    for attr in ("train_csv", "internal_csv", "external_csv", "clinical_csv"):
        value = getattr(args, attr)
        if value is not None:
            setattr(paths, attr, value)

    summary = run(paths)

    print("\nSelected spatial features:", summary["shs_features"])
    print("Selected clinical features:", summary["clinical_features"])
    print("\nPerformance (C-index with 95% CI):")
    for model_name, perf in summary["performance"].items():
        for cohort, m in perf.items():
            print(
                f"  {model_name:<8} {cohort:<14} "
                f"{m['c_index']:.3f} ({m['ci_lower']:.3f}-{m['ci_upper']:.3f})"
            )
    print(f"\nOutputs written to: {paths.output_dir}")


if __name__ == "__main__":
    main()
