"""Serialization helpers for generated datasets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eqnn.datasets.heisenberg import DatasetBundle, DatasetSplit


def save_dataset_split(split: DatasetSplit, output_path: str | Path) -> None:
    output_path = Path(output_path)
    np.savez_compressed(
        output_path,
        states=split.states,
        labels=split.labels,
        coupling_ratios=split.coupling_ratios,
        ground_state_energies=split.ground_state_energies,
    )


def load_dataset_split(input_path: str | Path) -> DatasetSplit:
    input_path = Path(input_path)
    with np.load(input_path, allow_pickle=False) as data:
        return DatasetSplit(
            states=data["states"],
            labels=data["labels"],
            coupling_ratios=data["coupling_ratios"],
            ground_state_energies=data["ground_state_energies"],
        )


def save_dataset_bundle(bundle: DatasetBundle, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_dataset_split(bundle.train, output_dir / "train.npz")
    save_dataset_split(bundle.test, output_dir / "test.npz")
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(bundle.metadata, indent=2, sort_keys=True) + "\n")


def load_dataset_bundle(output_dir: str | Path) -> DatasetBundle:
    output_dir = Path(output_dir)
    metadata_path = output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    return DatasetBundle(
        train=load_dataset_split(output_dir / "train.npz"),
        test=load_dataset_split(output_dir / "test.npz"),
        metadata=metadata,
    )
