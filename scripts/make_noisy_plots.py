#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_line_plot(df: pd.DataFrame, title: str, x: str, y: str, yerr: str | None, outpath: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    for label, g in df.groupby("series_label"):
        g = g.sort_values(by=x)
        if yerr and yerr in g.columns and g[yerr].notna().any():
            plt.errorbar(g[x], g[y], yerr=g[yerr], marker="o", label=label)
        else:
            plt.plot(g[x], g[y], marker="o", label=label)
    plt.xlabel(x.replace("_", " ").title())
    plt.ylabel(y.replace("_", " ").title())
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_heatmap(table: pd.DataFrame, title: str, outpath: Path) -> None:
    plt.figure(figsize=(8, 4.8))
    plt.imshow(table.values, aspect="auto")
    plt.xticks(range(len(table.columns)), [str(c) for c in table.columns], rotation=45, ha="right")
    plt.yticks(range(len(table.index)), [str(i) for i in table.index])
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_train_size_noise_family_plot(df: pd.DataFrame, title: str, outpath: Path) -> None:
    plt.figure(figsize=(9, 5.5))

    noise_levels = sorted(df["noise_strength"].dropna().unique())

    su2_shades = ["#08306b", "#08519c", "#2171b5", "#4292c6", "#6baed6", "#9ecae1"]
    hea_shades = ["#7f2704", "#a63603", "#d94801", "#f16913", "#fd8d3c", "#fdae6b"]

    color_map = {
        "su2_qcnn": su2_shades,
        "hea_qcnn": hea_shades,
    }
    marker_map = {
        "su2_qcnn": "o",
        "hea_qcnn": "s",
    }

    plotted_any = False

    for model_family in ["su2_qcnn", "hea_qcnn"]:
        model_df = df[df["model_family"] == model_family].copy()
        if model_df.empty:
            continue

        shades = color_map[model_family]

        for i, noise_strength in enumerate(noise_levels):
            sub = model_df[
                model_df["noise_strength"].round(6) == round(float(noise_strength), 6)
            ].copy()
            if sub.empty:
                continue

            sub = sub.sort_values(by="train_size")
            color = shades[min(i, len(shades) - 1)]
            line_style = "-" if sub["seeds_present"].min() >= 3 else "--"
            label = f"{model_family}, noise={noise_strength:g}"
            plotted_any = True

            if "test_accuracy_std" in sub.columns and sub["test_accuracy_std"].notna().any():
                plt.errorbar(
                    sub["train_size"],
                    sub["test_accuracy_mean"],
                    yerr=sub["test_accuracy_std"],
                    marker=marker_map[model_family],
                    color=color,
                    linestyle=line_style,
                    label=label,
                )
            else:
                plt.plot(
                    sub["train_size"],
                    sub["test_accuracy_mean"],
                    marker=marker_map[model_family],
                    color=color,
                    linestyle=line_style,
                    label=label,
                )

    if not plotted_any:
        plt.close()
        return

    plt.xlabel("Train size")
    plt.ylabel("Test accuracy mean")
    plt.ylim(0.0, 1.05)
    plt.title(title)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_global_grid_plot(df: pd.DataFrame, outpath: Path) -> None:
    noise_models = ["depolarizing", "coherent_overrotation"]
    qubit_values = [5, 7, 9]

    su2_shades = ["#08306b", "#08519c", "#2171b5", "#4292c6", "#6baed6", "#9ecae1"]
    hea_shades = ["#7f2704", "#a63603", "#d94801", "#f16913", "#fd8d3c", "#fdae6b"]

    color_map = {
        "su2_qcnn": su2_shades,
        "hea_qcnn": hea_shades,
    }
    marker_map = {
        "su2_qcnn": "o",
        "hea_qcnn": "s",
    }

    noise_levels = sorted(df["noise_strength"].dropna().unique())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)

    legend_items = []

    for row_idx, noise_model in enumerate(noise_models):
        for col_idx, n in enumerate(qubit_values):
            ax = axes[row_idx, col_idx]
            subpanel = df[
                (df["noise_model_name"] == noise_model) &
                (df["num_qubits"] == n)
            ].copy()

            if subpanel.empty:
                ax.set_title(f"n={n}, {noise_model}\n(no data)")
                ax.set_xlabel("Train size")
                ax.set_ylabel("Test accuracy mean")
                ax.set_ylim(0.0, 1.05)
                continue

            for model_family in ["su2_qcnn", "hea_qcnn"]:
                model_df = subpanel[subpanel["model_family"] == model_family].copy()
                if model_df.empty:
                    continue

                shades = color_map[model_family]

                for i, noise_strength in enumerate(noise_levels):
                    line_df = model_df[
                        model_df["noise_strength"].round(6) == round(float(noise_strength), 6)
                    ].copy()
                    if line_df.empty:
                        continue

                    line_df = line_df.sort_values(by="train_size")
                    color = shades[min(i, len(shades) - 1)]
                    linestyle = "-" if line_df["seeds_present"].min() >= 3 else "--"
                    label = f"{model_family}, noise={noise_strength:g}"

                    if line_df["test_accuracy_std"].notna().any():
                        container = ax.errorbar(
                            line_df["train_size"],
                            line_df["test_accuracy_mean"],
                            yerr=line_df["test_accuracy_std"],
                            marker=marker_map[model_family],
                            color=color,
                            linestyle=linestyle,
                            label=label,
                        )
                        handle = container.lines[0]
                    else:
                        line = ax.plot(
                            line_df["train_size"],
                            line_df["test_accuracy_mean"],
                            marker=marker_map[model_family],
                            color=color,
                            linestyle=linestyle,
                            label=label,
                        )[0]
                        handle = line

                    legend_items.append((label, handle))

            title = f"n={n}, {noise_model}"
            if n == 9:
                title += "\n(best available; HEA incomplete)"
            ax.set_title(title)
            ax.set_xlabel("Train size")
            ax.set_ylabel("Test accuracy mean")
            ax.set_ylim(0.0, 1.05)

    seen = set()
    uniq_handles = []
    uniq_labels = []
    for label, handle in legend_items:
        if label not in seen:
            seen.add(label)
            uniq_labels.append(label)
            uniq_handles.append(handle)

    if uniq_handles:
        fig.legend(
            uniq_handles,
            uniq_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=4,
            fontsize=8,
            frameon=True,
        )

    fig.suptitle(
        "Qiskit mixed odd runs: test accuracy vs train size across n and noise models",
        fontsize=14,
        y=1.08,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate plots from qiskit_mixed_odd aggregated summary output.")
    ap.add_argument("--aggregated-csv", required=True, help="Path to qiskit_mixed_odd_aggregated_summary.csv")
    ap.add_argument("--output-dir", required=True, help="Directory to write plots")
    args = ap.parse_args()

    aggregated_csv = Path(args.aggregated_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(aggregated_csv)

    for col in [
        "num_qubits",
        "train_size",
        "epochs",
        "noise_strength",
        "seeds_present",
        "test_accuracy_mean",
        "test_accuracy_std",
        "test_loss_mean",
        "test_loss_std",
        "history_best_accuracy_mean",
        "history_best_accuracy_std",
        "train_time_sec_mean",
        "total_time_sec_mean",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    complete = df[df["seeds_present"] >= 3].copy()
    available = df[df["seeds_present"] >= 1].copy()

    # 1. Accuracy vs noise strength at train_size=12 for n=5,7.
    for n in [5, 7]:
        for noise_model in sorted(complete["noise_model_name"].dropna().unique()):
            sub = complete[
                (complete["num_qubits"] == n)
                & (complete["noise_model_name"] == noise_model)
                & (complete["train_size"] == 12)
            ].copy()
            if sub.empty:
                continue
            sub["series_label"] = sub["model_family"]
            save_line_plot(
                sub,
                title=f"n={n}, train_size=12, {noise_model}: test accuracy vs noise strength",
                x="noise_strength",
                y="test_accuracy_mean",
                yerr="test_accuracy_std",
                outpath=output_dir / f"accuracy_vs_noise_n{n}_{noise_model}_train12.png",
            )

    # 2. n=7 train-size sensitivity at representative noise strengths.
    for noise_model in sorted(complete["noise_model_name"].dropna().unique()):
        for strength in [0.0, 0.01, 0.1]:
            sub = complete[
                (complete["num_qubits"] == 7)
                & (complete["noise_model_name"] == noise_model)
                & (complete["noise_strength"].round(6) == round(strength, 6))
            ].copy()
            if sub.empty:
                continue
            sub["series_label"] = sub["model_family"]
            save_line_plot(
                sub,
                title=f"n=7, {noise_model}, noise={strength}: test accuracy vs train size",
                x="train_size",
                y="test_accuracy_mean",
                yerr="test_accuracy_std",
                outpath=output_dir / f"accuracy_vs_train_size_n7_{noise_model}_noise_{str(strength).replace('.', 'p')}.png",
            )

    # 3. n=9 extension at train_size=12 where complete data exist.
    n9 = complete[complete["num_qubits"] == 9].copy()
    if not n9.empty:
        for noise_model in sorted(n9["noise_model_name"].dropna().unique()):
            sub = n9[
                (n9["noise_model_name"] == noise_model)
                & (n9["train_size"] == 12)
            ].copy()
            if sub.empty:
                continue
            sub["series_label"] = sub["model_family"]
            save_line_plot(
                sub,
                title=f"n=9, train_size=12, {noise_model}: test accuracy vs noise strength",
                x="noise_strength",
                y="test_accuracy_mean",
                yerr="test_accuracy_std",
                outpath=output_dir / f"accuracy_vs_noise_n9_{noise_model}_train12.png",
            )

    # 4. Coverage heatmaps for n=9.
    n9_all = df[df["num_qubits"] == 9].copy()
    if not n9_all.empty:
        for noise_model in sorted(n9_all["noise_model_name"].dropna().unique()):
            sub = n9_all[n9_all["noise_model_name"] == noise_model].copy()
            if sub.empty:
                continue
            sub["row"] = sub["model_family"] + "_train" + sub["train_size"].astype(int).astype(str)
            pivot = sub.pivot_table(
                index="row",
                columns="noise_strength",
                values="seeds_present",
                aggfunc="max",
            ).sort_index()
            save_heatmap(
                pivot,
                title=f"n=9 coverage heatmap: {noise_model}",
                outpath=output_dir / f"coverage_heatmap_n9_{noise_model}.png",
            )

    # 5. Per-panel combined plots, including partial n=9.
    for n in [5, 7, 9]:
        for noise_model in sorted(available["noise_model_name"].dropna().unique()):
            sub = available[
                (available["num_qubits"] == n)
                & (available["noise_model_name"] == noise_model)
            ].copy()

            if sub.empty:
                continue

            title = f"Test accuracy vs train size: n={n}, {noise_model}"
            if n == 9:
                title += " (best available; HEA incomplete)"

            save_train_size_noise_family_plot(
                sub,
                title=title,
                outpath=output_dir / f"train_size_accuracy_n{n}_{noise_model}_all_available.png",
            )

    # 6. Global 2x3 grid plot.
    save_global_grid_plot(
        available,
        output_dir / "train_size_accuracy_global_grid_all_available.png",
    )

    # 7. Optional runtime plot if timing becomes populated later.
    if "total_time_sec_mean" in complete.columns and complete["total_time_sec_mean"].notna().any():
        runtime_sub = complete[
            (complete["train_size"] == 12)
            & (complete["noise_strength"] == 0.01)
        ].copy()
        if not runtime_sub.empty:
            runtime_sub["series_label"] = runtime_sub["model_family"] + " | " + runtime_sub["noise_model_name"]
            save_line_plot(
                runtime_sub,
                title="Runtime vs qubit count at train_size=12, noise=0.01",
                x="num_qubits",
                y="total_time_sec_mean",
                yerr=None,
                outpath=output_dir / "runtime_vs_qubits_train12_noise_0p01.png",
            )

    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()