import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def require_env(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing environment variable: {name}")
    path = Path(value)

    if not path.exists():
        raise SystemExit(f"{name} does not exist: {path}")
    return path

global_root = require_env("GLOBAL_ROOT")
selected_root = require_env("SELECTED_ROOT")
sym_root = require_env("SYM_ROOT")
outdir = Path("data/noisy_comparisons/n7_depolarizing_diagnostic_results")
outdir.mkdir(parents=True, exist_ok=True)

global_df = pd.read_csv(global_root / "combined_summary.csv")
selected_df = pd.read_csv(selected_root / "combined_summary.csv")
sym_df = pd.read_csv(sym_root / "combined_summary.csv")

for df in [global_df, selected_df, sym_df]:

    for col in [
        "noise_strength",
        "train_size",
        "mean_test_accuracy",
        "mean_train_accuracy",
        "mean_test_loss",
        "mean_train_loss",
        "noisy_qubit_index",
        "mean_test_equivariance_error_mean",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print("Wrote", path)

# ---------------------------------------------------------------------
# Plot 1: fine-grained global depolarizing curves by train size
# ---------------------------------------------------------------------

for train_size in sorted(global_df["train_size"].dropna().unique()):
    sub = global_df[global_df["train_size"] == train_size]
    plt.figure(figsize=(8, 5))
    for model, g in sub.groupby("model_family"):
        g = g.sort_values("noise_strength")
        plt.plot(
            g["noise_strength"],
            g["mean_test_accuracy"],
            marker="o",
            label=f"{model} test",
        )

        if "mean_train_accuracy" in g.columns:
            plt.plot(
                g["noise_strength"],
                g["mean_train_accuracy"],
                marker="x",
                linestyle="--",
                label=f"{model} train",
            )

    plt.xlabel("Depolarizing noise strength")
    plt.ylabel("Mean accuracy")
    plt.title(f"n=7 global depolarizing fine sweep, train_size={int(train_size)}")
    plt.grid(True, alpha=0.3)
    plt.legend()

    savefig(outdir / f"global_depolarizing_n7_train{int(train_size)}.png")

# ---------------------------------------------------------------------
# Plot 2: delta from noise 0 to 0.1
# ---------------------------------------------------------------------

base = global_df[global_df["noise_strength"].round(6) == 0.0].copy()
high = global_df[global_df["noise_strength"].round(6) == 0.1].copy()
keys = ["model_family", "train_size"]

merged = high.merge(
    base[keys + ["mean_test_accuracy"]],
    on=keys,
    suffixes=("_noise_0p1", "_noise_0"),
)

merged["delta_0p1_minus_0"] = (
    merged["mean_test_accuracy_noise_0p1"] - merged["mean_test_accuracy_noise_0"]

)

delta_path = outdir / "global_depolarizing_delta_0p1_vs_0.csv"
merged.sort_values(["model_family", "train_size"]).to_csv(delta_path, index=False)

print("Wrote", delta_path)
print(
    merged.sort_values("delta_0p1_minus_0")[
        [
            "model_family",
            "train_size",
            "mean_test_accuracy_noise_0",
            "mean_test_accuracy_noise_0p1",
            "delta_0p1_minus_0",
        ]
    ].to_string(index=False)
)

# ---------------------------------------------------------------------
# Plot 3: selected-qubit depolarizing sensitivity
# ---------------------------------------------------------------------

for train_size in sorted(selected_df["train_size"].dropna().unique()):
    for strength in sorted(selected_df["noise_strength"].dropna().unique()):
        sub = selected_df[
            (selected_df["train_size"] == train_size)
            & (selected_df["noise_strength"] == strength)
        ]
        if sub.empty:
            continue

        plt.figure(figsize=(8, 5))

        for model, g in sub.groupby("model_family"):
            g = g.sort_values("noisy_qubit_index")
            plt.plot(
                g["noisy_qubit_index"],
                g["mean_test_accuracy"],
                marker="o",
                label=model,
            )

        plt.xlabel("Noisy qubit index")
        plt.ylabel("Mean test accuracy")
        plt.title(
            f"n=7 selected-qubit depolarizing, "
            f"train_size={int(train_size)}, noise={strength:g}"
        )

        plt.grid(True, alpha=0.3)
        plt.legend()
        strength_label = str(strength).replace(".", "p")
        savefig(
            outdir
            / f"selected_depolarizing_n7_train{int(train_size)}_noise{strength_label}.png"
        )
# ---------------------------------------------------------------------
# Plot 4: selected-qubit collapsed mean/min/max bands
# ---------------------------------------------------------------------

selected_collapsed = (
    selected_df
    .groupby(
        ["model_family", "train_size", "noise_strength"],
        dropna=False,
    )
    .agg(
        mean_test_accuracy=("mean_test_accuracy", "mean"),
        min_test_accuracy=("mean_test_accuracy", "min"),
        max_test_accuracy=("mean_test_accuracy", "max"),
        qubits_swept=("noisy_qubit_index", "nunique"),
    )
    .reset_index()
)

collapsed_path = outdir / "selected_depolarizing_collapsed_summary.csv"
selected_collapsed.to_csv(collapsed_path, index=False)
print("Wrote", collapsed_path)

for train_size in sorted(selected_collapsed["train_size"].dropna().unique()):
    sub = selected_collapsed[selected_collapsed["train_size"] == train_size]
    plt.figure(figsize=(8, 5))

    for model, g in sub.groupby("model_family"):
        g = g.sort_values("noise_strength")
        plt.plot(
            g["noise_strength"],
            g["mean_test_accuracy"],
            marker="o",
            label=f"{model} mean over qubits",
        )

        plt.fill_between(
            g["noise_strength"],
            g["min_test_accuracy"],
            g["max_test_accuracy"],
            alpha=0.15,
            label=f"{model} min-max over qubits",
        )

    plt.xlabel("Depolarizing noise strength")
    plt.ylabel("Mean test accuracy")
    plt.title(f"n=7 selected-qubit depolarizing collapsed, train_size={int(train_size)}")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)

    savefig(outdir / f"selected_depolarizing_collapsed_n7_train{int(train_size)}.png")

# ---------------------------------------------------------------------
# Plot 5: symmetry drift diagnostic
# ---------------------------------------------------------------------

if "mean_test_equivariance_error_mean" in sym_df.columns:

    plt.figure(figsize=(8, 5))

    for model, g in sym_df.groupby("model_family"):

        g = g.sort_values("noise_strength")

        plt.plot(

            g["noise_strength"],

            g["mean_test_equivariance_error_mean"],

            marker="o",

            label=model,

        )

    plt.xlabel("Depolarizing noise strength")

    plt.ylabel("Mean empirical symmetry prediction drift")

    plt.title("n=7 depolarizing empirical symmetry diagnostic")

    plt.grid(True, alpha=0.3)

    plt.legend()

    savefig(outdir / "symmetry_drift_vs_depolarizing_noise_n7.png")

# ---------------------------------------------------------------------

# Plot 6: accuracy and symmetry drift side by side in CSV form

# ---------------------------------------------------------------------

sym_cols = [

    "model_family",

    "train_size",

    "noise_strength",

    "mean_test_accuracy",

    "mean_train_accuracy",

    "mean_test_equivariance_error_mean",

    "num_runs",

]

sym_cols = [c for c in sym_cols if c in sym_df.columns]

sym_out = outdir / "symmetry_diagnostic_summary.csv"

sym_df[sym_cols].sort_values(["model_family", "noise_strength"]).to_csv(

    sym_out,

    index=False,

)

print("Wrote", sym_out)

print("Results written to:", outdir)