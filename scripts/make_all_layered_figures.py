import os
import re
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def resolve_out_root() -> Path:
    """
    Resolve output root in priority order:
    1. First command-line argument, if provided.
    2. OUT_ROOT environment variable, if set.
    3. Latest qiskit_mixed_odd_n5n7_expanded_<numeric_id> directory under
       data/noisy_comparisons, choosing the largest numeric suffix.
    """
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser().resolve()

    env_root = os.environ.get("OUT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    base = Path("data/noisy_comparisons")
    pattern = re.compile(r"qiskit_mixed_odd_n5n7_expanded_(\d{8,})$")

    candidates = []
    for p in base.glob("qiskit_mixed_odd_n5n7_expanded_*"):
        if not p.is_dir():
            continue
        match = pattern.search(p.name)
        if match:
            candidates.append((int(match.group(1)), p))

    if not candidates:
        raise FileNotFoundError(
            "Could not resolve output root. Provide a directory argument, set OUT_ROOT, "
            "or run from the repo root with a matching directory under "
            "data/noisy_comparisons/qiskit_mixed_odd_n5n7_expanded_<id>."
        )

    _, latest = max(candidates, key=lambda item: item[0])
    return latest.resolve()


out_root = resolve_out_root()
summary = out_root / "combined_summary.csv"

if not summary.exists():
    raise FileNotFoundError(f"Could not find combined summary CSV: {summary}")

print(f"Using OUT_ROOT: {out_root}")
print(f"Using summary: {summary}")

df = pd.read_csv(summary)

figdir = out_root / "results" / "figures_layered"
figdir.mkdir(parents=True, exist_ok=True)

for col in [
    "noise_strength",
    "num_qubits",
    "train_size",
    "mean_test_accuracy",
    "noisy_qubit_index",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# ---------------------------------------------------------------------
# Plot settings
# ---------------------------------------------------------------------

# Main report focus. Change these if you want all n/train sizes.
N_VALUES = [7]
TRAIN_SIZES = [16]

# For selected-qubit layered plots.
NOISE_STRENGTHS = [0.001, 0.005, 0.01, 0.05, 0.1]

base_colors = {
    "su2_qcnn": "tab:blue",
    "hea_qcnn": "tab:orange",
}

model_styles = {
    "su2_qcnn": "-",
    "hea_qcnn": "--",
}


def blend_with_white(color, amount):
    """amount=0 gives original color; amount=1 gives white."""
    rgb = mcolors.to_rgb(color)
    return tuple((1 - amount) * c + amount for c in rgb)


def safe_strength_label(x):
    return str(x).replace(".", "p")


def savefig(name):
    path = figdir / name
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print("Wrote", path)


def noise_variant_label(row):
    noise = row["noise_model_name"]
    if noise == "coherent_overrotation":
        return f"coherent_{row.get('coherent_overrotation_mode', 'unknown')}"
    if row["noise_application_scope"] == "selected_qubits":
        return f"{noise}_selected_avg"
    return str(noise)


# ---------------------------------------------------------------------
# Prepare active and selected-qubit data
# ---------------------------------------------------------------------

active = df[df["noise_application_scope"] == "active"].copy()
if not active.empty:
    active["noise_variant"] = active.apply(noise_variant_label, axis=1)

selected = df[df["noise_application_scope"] == "selected_qubits"].copy()

selected_collapsed = pd.DataFrame()
if not selected.empty:
    selected_collapsed = (
        selected
        .groupby(
            [
                "noise_model_name",
                "num_qubits",
                "train_size",
                "model_family",
                "noise_strength",
            ],
            dropna=False,
        )
        .agg(
            mean_test_accuracy=("mean_test_accuracy", "mean"),
            min_test_accuracy=("mean_test_accuracy", "min"),
            max_test_accuracy=("mean_test_accuracy", "max"),
            std_test_accuracy=("mean_test_accuracy", "std"),
            num_qubits_swept=("noisy_qubit_index", "nunique"),
        )
        .reset_index()
    )
    selected_collapsed["noise_application_scope"] = "selected_qubits_collapsed"
    selected_collapsed["coherent_overrotation_mode"] = pd.NA
    selected_collapsed["noise_variant"] = (
        selected_collapsed["noise_model_name"] + "_selected_avg"
    )


# ---------------------------------------------------------------------
# 1. Layered selected-qubit plots by noise strength
# ---------------------------------------------------------------------

for n in N_VALUES:
    for train_size in TRAIN_SIZES:
        for noise_model in sorted(selected["noise_model_name"].dropna().unique()):
            sub = selected[
                (selected["num_qubits"] == n)
                & (selected["train_size"] == train_size)
                & (selected["noise_model_name"] == noise_model)
                & (selected["noise_strength"].isin(NOISE_STRENGTHS))
            ].copy()

            if sub.empty:
                continue

            strengths = sorted(sub["noise_strength"].dropna().unique())
            if not strengths:
                continue

            plt.figure(figsize=(9, 6))

            # Higher noise = darker line.
            max_idx = max(len(strengths) - 1, 1)

            for model in sorted(sub["model_family"].dropna().unique()):
                model_color = base_colors.get(model, "gray")
                model_style = model_styles.get(model, "-")

                for i, strength in enumerate(strengths):
                    g = sub[
                        (sub["model_family"] == model)
                        & (sub["noise_strength"] == strength)
                    ].sort_values("noisy_qubit_index")

                    if g.empty:
                        continue

                    # Light for low noise, dark for high noise.
                    whiten_amount = 0.70 - 0.55 * (i / max_idx)
                    color = blend_with_white(model_color, whiten_amount)

                    plt.plot(
                        g["noisy_qubit_index"],
                        g["mean_test_accuracy"],
                        marker="o",
                        linestyle=model_style,
                        linewidth=2,
                        color=color,
                        label=f"{model}, noise={strength:g}",
                    )

            plt.xlabel("Noisy qubit index")
            plt.ylabel("Mean test accuracy")
            plt.title(
                f"Selected-qubit sensitivity by noise strength\n"
                f"{noise_model}, n={n}, train_size={train_size}"
            )
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=8, ncol=2)

            savefig(
                f"layered_selected_qubit_{noise_model}_n{n}_train{train_size}.png"
            )


# ---------------------------------------------------------------------
# 2. Active/global noise curves layered by train size
# ---------------------------------------------------------------------

if not active.empty:
    for n in N_VALUES:
        for variant in sorted(active["noise_variant"].dropna().unique()):
            sub = active[
                (active["num_qubits"] == n)
                & (active["noise_variant"] == variant)
            ].copy()

            if sub.empty:
                continue

            train_sizes = sorted(sub["train_size"].dropna().unique())
            max_idx = max(len(train_sizes) - 1, 1)

            plt.figure(figsize=(9, 6))

            for model in sorted(sub["model_family"].dropna().unique()):
                for i, train_size in enumerate(train_sizes):
                    g = sub[
                        (sub["model_family"] == model)
                        & (sub["train_size"] == train_size)
                    ].sort_values("noise_strength")

                    if g.empty:
                        continue

                    whiten_amount = 0.70 - 0.55 * (i / max_idx)
                    color = blend_with_white(
                        base_colors.get(model, "gray"),
                        whiten_amount,
                    )

                    plt.plot(
                        g["noise_strength"],
                        g["mean_test_accuracy"],
                        marker="o",
                        linestyle=model_styles.get(model, "-"),
                        color=color,
                        linewidth=2,
                        label=f"{model}, train={int(train_size)}",
                    )

            plt.xlabel("Noise strength")
            plt.ylabel("Mean test accuracy")
            plt.title(f"Noise response by train size: {variant}, n={int(n)}")
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=8, ncol=2)

            savefig(f"active_by_train_size_{variant}_n{int(n)}.png")


# ---------------------------------------------------------------------
# 3. Active/global comparison across depolarizing + coherent modes
# ---------------------------------------------------------------------

if not active.empty:
    for n in N_VALUES:
        for train_size in TRAIN_SIZES:
            sub = active[
                (active["num_qubits"] == n)
                & (active["train_size"] == train_size)
            ].copy()

            if sub.empty:
                continue

            plt.figure(figsize=(10, 6))

            variants = sorted(sub["noise_variant"].dropna().unique())
            line_styles = ["-", "--", "-.", ":"]
            variant_style = {
                variant: line_styles[i % len(line_styles)]
                for i, variant in enumerate(variants)
            }

            for model in sorted(sub["model_family"].dropna().unique()):
                for variant in variants:
                    g = sub[
                        (sub["model_family"] == model)
                        & (sub["noise_variant"] == variant)
                    ].sort_values("noise_strength")

                    if g.empty:
                        continue

                    plt.plot(
                        g["noise_strength"],
                        g["mean_test_accuracy"],
                        marker="o",
                        linestyle=variant_style[variant],
                        color=base_colors.get(model, "gray"),
                        linewidth=2,
                        label=f"{model}, {variant}",
                    )

            plt.xlabel("Noise strength")
            plt.ylabel("Mean test accuracy")
            plt.title(
                f"Active/global noise comparison: "
                f"n={int(n)}, train_size={int(train_size)}"
            )
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=8, ncol=2)

            savefig(
                f"active_noise_model_comparison_n{int(n)}_train{int(train_size)}.png"
            )


# ---------------------------------------------------------------------
# 4. Selected-qubit collapsed mean with min-max bands
# ---------------------------------------------------------------------

if not selected_collapsed.empty:
    for n in N_VALUES:
        for train_size in TRAIN_SIZES:
            for noise_model in sorted(
                selected_collapsed["noise_model_name"].dropna().unique()
            ):
                sub = selected_collapsed[
                    (selected_collapsed["num_qubits"] == n)
                    & (selected_collapsed["train_size"] == train_size)
                    & (selected_collapsed["noise_model_name"] == noise_model)
                ].copy()

                if sub.empty:
                    continue

                plt.figure(figsize=(9, 6))

                for model in sorted(sub["model_family"].dropna().unique()):
                    g = sub[
                        sub["model_family"] == model
                    ].sort_values("noise_strength")

                    if g.empty:
                        continue

                    color = base_colors.get(model, "gray")

                    plt.plot(
                        g["noise_strength"],
                        g["mean_test_accuracy"],
                        marker="o",
                        linestyle=model_styles.get(model, "-"),
                        color=color,
                        linewidth=2,
                        label=f"{model} mean over noisy qubits",
                    )

                    plt.fill_between(
                        g["noise_strength"],
                        g["min_test_accuracy"],
                        g["max_test_accuracy"],
                        color=color,
                        alpha=0.15,
                        label=f"{model} min-max over qubits",
                    )

                plt.xlabel("Noise strength")
                plt.ylabel("Mean test accuracy")
                plt.title(
                    f"Selected-qubit collapsed sensitivity: {noise_model}, "
                    f"n={int(n)}, train_size={int(train_size)}"
                )
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=8)

                savefig(
                    f"selected_collapsed_band_{noise_model}_n{int(n)}_train{int(train_size)}.png"
                )


# ---------------------------------------------------------------------
# 5. High-noise bar plot with all noise variants together
# ---------------------------------------------------------------------

active_high = pd.DataFrame()
if not active.empty:
    active_high = active[active["noise_strength"].round(6) == 0.1].copy()
    active_high = active_high[
        [
            "noise_variant",
            "num_qubits",
            "train_size",
            "model_family",
            "mean_test_accuracy",
        ]
    ].copy()

selected_high = pd.DataFrame()
if not selected_collapsed.empty:
    selected_high = selected_collapsed[
        selected_collapsed["noise_strength"].round(6) == 0.1
    ][
        [
            "noise_variant",
            "num_qubits",
            "train_size",
            "model_family",
            "mean_test_accuracy",
            "min_test_accuracy",
            "max_test_accuracy",
        ]
    ].copy()

bar_df = pd.concat([active_high, selected_high], ignore_index=True)

if not bar_df.empty:
    for n in N_VALUES:
        for train_size in TRAIN_SIZES:
            sub = bar_df[
                (bar_df["num_qubits"] == n)
                & (bar_df["train_size"] == train_size)
            ].copy()

            if sub.empty:
                continue

            labels = sorted(sub["noise_variant"].dropna().unique())
            x = list(range(len(labels)))
            width = 0.35

            plt.figure(figsize=(12, 6))

            models = sorted(sub["model_family"].dropna().unique())
            for i, model in enumerate(models):
                vals = []
                yerr_low = []
                yerr_high = []

                for label in labels:
                    row = sub[
                        (sub["noise_variant"] == label)
                        & (sub["model_family"] == model)
                    ]

                    if row.empty:
                        vals.append(float("nan"))
                        yerr_low.append(0.0)
                        yerr_high.append(0.0)
                        continue

                    val = row["mean_test_accuracy"].iloc[0]
                    vals.append(val)

                    has_band = (
                        "min_test_accuracy" in row.columns
                        and "max_test_accuracy" in row.columns
                        and pd.notna(row["min_test_accuracy"].iloc[0])
                        and pd.notna(row["max_test_accuracy"].iloc[0])
                    )

                    if has_band:
                        min_val = row["min_test_accuracy"].iloc[0]
                        max_val = row["max_test_accuracy"].iloc[0]
                        yerr_low.append(max(0.0, val - min_val))
                        yerr_high.append(max(0.0, max_val - val))
                    else:
                        yerr_low.append(0.0)
                        yerr_high.append(0.0)

                offset = (i - (len(models) - 1) / 2) * width

                plt.bar(
                    [xx + offset for xx in x],
                    vals,
                    width=width,
                    color=base_colors.get(model, None),
                    alpha=0.8,
                    label=model,
                    yerr=[yerr_low, yerr_high],
                    capsize=3,
                )

            plt.xticks(x, labels, rotation=35, ha="right")
            plt.ylabel("Mean test accuracy")
            plt.title(
                f"All noise variants at noise=0.1: "
                f"n={int(n)}, train_size={int(train_size)}"
            )
            plt.grid(True, axis="y", alpha=0.3)
            plt.legend()

            savefig(
                f"all_noise_variants_high_noise_n{int(n)}_train{int(train_size)}.png"
            )


print("All layered figures written to:", figdir)