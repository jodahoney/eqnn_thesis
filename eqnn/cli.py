"""Command-line interface for repository utilities."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from eqnn.datasets.heisenberg import HeisenbergDatasetConfig, generate_dataset
from eqnn.datasets.io import save_dataset_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eqnn",
        description="Utilities for building and testing the EQNN simulator.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset_parser = subparsers.add_parser(
        "generate-dataset",
        help="Generate a bond-alternating Heisenberg ground-state dataset.",
    )
    dataset_parser.add_argument("--num-qubits", type=int, required=True)
    dataset_parser.add_argument("--ratio-min", type=float, default=0.4)
    dataset_parser.add_argument("--ratio-max", type=float, default=1.6)
    dataset_parser.add_argument("--num-points", type=int, default=31)
    dataset_parser.add_argument("--train-fraction", type=float, default=0.8)
    dataset_parser.add_argument("--critical-ratio", type=float, default=1.0)
    dataset_parser.add_argument("--exclusion-window", type=float, default=0.05)
    dataset_parser.add_argument(
        "--boundary",
        choices=("open", "periodic"),
        default="open",
    )
    dataset_parser.add_argument(
        "--eigensolver",
        choices=("auto", "dense", "sparse"),
        default="auto",
    )
    dataset_parser.add_argument("--split-seed", type=int, default=0)
    dataset_parser.add_argument("--output-dir", type=Path, required=True)
    dataset_parser.set_defaults(handler=_handle_generate_dataset)

    return parser


def _handle_generate_dataset(args: argparse.Namespace) -> int:
    config = HeisenbergDatasetConfig(
        num_qubits=args.num_qubits,
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
        num_points=args.num_points,
        train_fraction=args.train_fraction,
        critical_ratio=args.critical_ratio,
        exclusion_window=args.exclusion_window,
        boundary=args.boundary,
        eigensolver=args.eigensolver,
        split_seed=args.split_seed,
    )
    bundle = generate_dataset(config)
    save_dataset_bundle(bundle, args.output_dir)

    print(f"Wrote dataset to {args.output_dir.resolve()}")
    print(
        "Train samples: "
        f"{len(bundle.train)} | Test samples: {len(bundle.test)} | "
        f"State dimension: {bundle.train.states.shape[1]}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
