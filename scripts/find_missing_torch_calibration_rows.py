#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

SUMMARY_CSV = Path("/scratch/users/jdehoney/eqnn_thesis/data/comparisons/eqnn_vs_hea_purestate_50ep_torch_20529347/summary.csv")

MODEL_FAMILIES = ("su2_qcnn", "hea_qcnn")
NUM_QUBITS = (6, 7, 8, 9, 10, 11, 12, 13)
TRAIN_SIZES = (2, 4, 6, 8, 10, 12)
EPOCHS = 50
BACKEND = "torch_pure"

def main() -> None:
    observed: set[tuple[str, str, int, int, int]] = set()

    with SUMMARY_CSV.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            observed.add(
                (
                    row["backend_name"],
                    row["model_family"],
                    int(row["num_qubits"]),
                    int(row["train_size"]),
                    int(row["epochs"]),
                )
            )

    missing: list[tuple[str, int, int, int]] = []
    for model_family in MODEL_FAMILIES:
        for num_qubits in NUM_QUBITS:
            for train_size in TRAIN_SIZES:
                key = (BACKEND, model_family, num_qubits, train_size, EPOCHS)
                if key not in observed:
                    missing.append((model_family, num_qubits, train_size, EPOCHS))

    print(f"Found {len(missing)} missing summary rows for backend={BACKEND}")
    for model_family, num_qubits, train_size, epochs in missing:
        print(f"{model_family},{num_qubits},{train_size},{epochs}")

if __name__ == "__main__":
    main()