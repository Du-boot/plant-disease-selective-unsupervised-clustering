#!/usr/bin/env python3
"""Create publication-oriented figures for the Protocol A manuscript."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


REPO_ROOT = Path(__file__).resolve().parents[1]
if (REPO_ROOT / "results" / "followup_results_seedlevel" / "clean_main.csv").exists():
    ROOT = REPO_ROOT
    FOLLOWUP = ROOT / "results" / "followup_results_seedlevel"
    PATCH = ROOT / "results"
    OUT = ROOT / "outputs" / "manuscript_figures"
elif Path(r"F:\hospitol").exists():
    ROOT = Path(r"F:\hospitol")
    FOLLOWUP = ROOT / "protocolA_followup_20260720" / "followup_results"
    PATCH = ROOT / "protocolA_final_required_patch_20260720"
    OUT = ROOT / "protocolA_manuscript_figures_20260720"
else:
    ROOT = Path("/data1/D/hostal")
    FOLLOWUP = ROOT / "protocolA_followup_20260720" / "followup_results"
    PATCH = ROOT / "protocolA_final_required_patch_20260720"
    OUT = ROOT / "protocolA_manuscript_figures_20260720"
OUT.mkdir(parents=True, exist_ok=True)

DATASET_LABELS = {
    "F_new": "Fruit",
    "V_new": "Vegetable",
    "M_new_drop5_drop7": "Multi-crop",
    "G_new": "Auxiliary",
}

PALETTE = {
    "blue": "#4C78A8",
    "teal": "#54A24B",
    "orange": "#F58518",
    "red": "#E45756",
    "purple": "#8E6BBE",
    "gray": "#6B7280",
    "light": "#E8EDF3",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
})


def save(fig: plt.Figure, name: str) -> None:
    for ext in ["svg", "pdf"]:
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def pct(x):
    return np.asarray(x, dtype=float) * 100


def mean_std(df: pd.DataFrame, keys: list[str], value: str) -> pd.DataFrame:
    return df.groupby(keys)[value].agg(["mean", "std"]).reset_index()


def ensure_acc_kept(df: pd.DataFrame) -> pd.DataFrame:
    if "acc_kept" not in df.columns and "kept_error" in df.columns:
        df = df.copy()
        df["acc_kept"] = 1 - df["kept_error"].astype(float)
    return df


def fig1_workflow() -> None:
    labels = [
        "Leaf images",
        "ConvNeXt\n2048-D features",
        "UMAP\n100-D",
        "KMeans\nBirch\nAgglomerative",
        "Hungarian\ncluster alignment",
        "any2 / all3\nconsensus",
        "Post-hoc\nmany-to-one evaluation",
    ]
    fig, ax = plt.subplots(figsize=(7.2, 1.8))
    ax.set_axis_off()
    x0, y, w, h, gap = 0.02, 0.35, 0.12, 0.38, 0.025
    for i, lab in enumerate(labels):
        x = x0 + i * (w + gap)
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            linewidth=0.9,
            facecolor="#F8FAFC" if i not in {5, 6} else "#EEF6F3",
            edgecolor="#6B7280" if i not in {5, 6} else "#256F68",
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, lab, ha="center", va="center", fontsize=7)
        if i < len(labels) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.003, y + h / 2), (x + w + gap - 0.004, y + h / 2),
                                         arrowstyle="-|>", mutation_scale=9, linewidth=0.8, color="#374151"))
    ax.text(0.5, 0.12, "Disease labels are used only after all clustering and rejection decisions are fixed.",
            ha="center", va="center", fontsize=7, color="#4B5563")
    save(fig, "fig1_protocolA_workflow")


def fig2_main_results() -> None:
    df = ensure_acc_kept(pd.read_csv(FOLLOWUP / "clean_main.csv"))
    df = df[(df["dataset"].isin(["F_new", "V_new", "M_new_drop5_drop7"])) & (df["method"] == "all3")]
    rows = []
    for ds, sub in df.groupby("dataset"):
        rows.append({
            "dataset": DATASET_LABELS[ds],
            "acc_kept": sub["acc_kept"].mean(),
            "acc_kept_std": sub["acc_kept"].std(ddof=0),
            "coverage": (1 - sub["rejection_rate"]).mean(),
            "coverage_std": sub["rejection_rate"].std(ddof=0),
            "overall": sub["overall_accuracy"].mean(),
            "overall_std": sub["overall_accuracy"].std(ddof=0),
        })
    d = pd.DataFrame(rows)
    metrics = [("acc_kept", "acc$_{kept}$", PALETTE["blue"]), ("coverage", "coverage", PALETTE["teal"]), ("overall", "overall", PALETTE["orange"])]
    x = np.arange(len(d))
    width = 0.24
    fig, ax = plt.subplots(figsize=(3.6, 2.35))
    for i, (key, lab, color) in enumerate(metrics):
        ax.bar(x + (i - 1) * width, pct(d[key]), width, yerr=pct(d[f"{key}_std"]), color=color, label=lab, capsize=2, linewidth=0)
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(d["dataset"])
    ax.set_ylim(0, 105)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title("Unified UMAP100 + K20 all3 setting", fontsize=8)
    save(fig, "fig2_main_results_all3")


def fig3_baseline_comparison() -> None:
    main = ensure_acc_kept(pd.read_csv(FOLLOWUP / "clean_main.csv"))
    rnd = ensure_acc_kept(pd.read_csv(FOLLOWUP / "random_same_coverage_completed.csv"))
    main = main[main["dataset"].isin(["F_new", "V_new", "M_new_drop5_drop7"])]
    rnd = rnd[rnd["dataset"].isin(["F_new", "V_new", "M_new_drop5_drop7"])]
    rnd = rnd.groupby(["dataset", "seed"]).agg({"acc_kept": "mean", "rejection_rate": "mean"}).reset_index()
    rnd["method"] = "random"
    use = pd.concat([
        main[main["method"].isin(["kmeans", "birch", "agg", "any2", "all3", "kmeans_distance_reject"])][["dataset", "seed", "method", "acc_kept", "rejection_rate"]],
        rnd[["dataset", "seed", "method", "acc_kept", "rejection_rate"]],
    ], ignore_index=True)
    methods = ["kmeans", "birch", "agg", "any2", "random", "kmeans_distance_reject", "all3"]
    colors = [PALETTE["gray"], "#9CA3AF", "#CBD5E1", PALETTE["teal"], PALETTE["purple"], PALETTE["orange"], PALETTE["blue"]]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), sharey=True)
    for ax, ds in zip(axes, ["F_new", "V_new", "M_new_drop5_drop7"]):
        sub = use[use["dataset"] == ds]
        vals = [sub[sub["method"] == m]["acc_kept"].mean() * 100 for m in methods]
        errs = [sub[sub["method"] == m]["acc_kept"].std(ddof=0) * 100 for m in methods]
        ax.bar(np.arange(len(methods)), vals, yerr=errs, color=colors, capsize=2)
        ax.set_title(DATASET_LABELS[ds], fontsize=8)
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(["KM", "Birch", "Agg", "any2", "rand", "dist.", "all3"], rotation=40, ha="right")
        ax.set_ylim(80, 101)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    axes[0].set_ylabel("acc$_{kept}$ (%)")
    save(fig, "fig3_baseline_acc_kept")


def fig4_risk_coverage() -> None:
    df = pd.read_csv(PATCH / "risk_coverage_discrete" / "risk_coverage_discrete_operating_points.csv")
    df = df[df["dataset"].isin(["F_new", "V_new", "M_new_drop5_drop7"])]
    methods = ["kmeans", "any2", "kmeans_distance_reject", "all3"]
    markers = {"kmeans": "o", "any2": "s", "kmeans_distance_reject": "^", "all3": "D"}
    colors = {"F_new": PALETTE["blue"], "V_new": PALETTE["teal"], "M_new_drop5_drop7": PALETTE["orange"]}
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    for ds, sub in df.groupby("dataset"):
        for m in methods:
            r = sub[sub["method"] == m]
            if r.empty:
                continue
            ax.scatter(r["coverage"] * 100, r["selective_risk"] * 100, s=32, marker=markers[m],
                       color=colors[ds], edgecolor="white", linewidth=0.5)
        ax.plot(sub.set_index("method").loc[methods]["coverage"] * 100,
                sub.set_index("method").loc[methods]["selective_risk"] * 100,
                color=colors[ds], linewidth=0.8, alpha=0.7, label=DATASET_LABELS[ds])
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Selective risk, 1 - acc$_{kept}$ (%)")
    ax.set_title("Discrete risk-coverage operating points", fontsize=8)
    ax.legend(loc="upper left")
    ax.grid(color="#E5E7EB", linewidth=0.6)
    save(fig, "fig4_discrete_risk_coverage")


def fig5_parameter_heatmap() -> None:
    df = ensure_acc_kept(pd.read_csv(FOLLOWUP / "parameter_sensitivity_5seeds.csv"))
    df = df[(df["method"] == "all3") & (df["dataset"].isin(["V_new", "M_new_drop5_drop7"]))]
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.45), constrained_layout=True)
    for ax, ds in zip(axes, ["V_new", "M_new_drop5_drop7"]):
        sub = df[df["dataset"] == ds].groupby(["dim", "k"])["acc_kept"].mean().reset_index()
        piv = sub.pivot(index="dim", columns="k", values="acc_kept").sort_index()
        im = ax.imshow(piv.values * 100, aspect="auto", cmap="YlGnBu", vmin=80, vmax=100)
        ax.set_title(DATASET_LABELS[ds], fontsize=8)
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("UMAP dimensions")
        ax.set_xticks(np.arange(len(piv.columns)))
        ax.set_xticklabels([str(c) for c in piv.columns])
        ax.set_yticks(np.arange(len(piv.index)))
        ax.set_yticklabels([str(i) for i in piv.index])
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                if np.isfinite(piv.values[i, j]):
                    ax.text(j, i, f"{piv.values[i, j] * 100:.1f}", ha="center", va="center", fontsize=5,
                            color="black" if piv.values[i, j] < 0.95 else "white")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88, pad=0.02)
    cbar.set_label("acc$_{kept}$ (%)")
    save(fig, "fig5_parameter_sensitivity_heatmap")


def fig6_weak_classes() -> None:
    df = pd.read_csv(PATCH / "class_level_5seed" / "weak_or_high_rejection_classes_5seed.csv")
    df = df[df["dataset"].isin(["V_new", "M_new_drop5_drop7"])].head(10).iloc[::-1]
    labels = [f"{DATASET_LABELS[d]}: {c}" for d, c in zip(df["dataset"], df["class_name"])]
    fig, ax = plt.subplots(figsize=(5.8, 3.1))
    y = np.arange(len(df))
    ax.barh(y - 0.18, df["acc_kept_mean"] * 100, height=0.34, color=PALETTE["blue"], label="acc$_{kept}$")
    ax.barh(y + 0.18, df["rejection_rate_mean"] * 100, height=0.34, color=PALETTE["red"], label="rejection")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5.5)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Percentage (%)")
    ax.legend(loc="lower right")
    ax.set_title("Class-level weak points across five seeds", fontsize=8)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.6)
    save(fig, "fig6_class_level_weak_points")


def read_matrix(path: Path):
    df = pd.read_csv(path)
    labels = df.iloc[:, 0].astype(str).tolist()
    mat = df.iloc[:, 1:].astype(float).values
    cols = df.columns[1:].tolist()
    return labels, cols, mat


def fig7_confusion_matrices() -> None:
    root = PATCH / "class_level_5seed"
    kept_labels, kept_cols, kept = read_matrix(root / "M_new_drop5_drop7_5seed_avg_kept_confusion_row_normalized.csv")
    all_labels, all_cols, allm = read_matrix(root / "M_new_drop5_drop7_5seed_avg_all_confusion_with_rejected_row_normalized.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    for ax, mat, title, cols, rows in [
        (axes[0], kept, "Retained samples", kept_cols, kept_labels),
        (axes[1], allm, "All samples with Rejected", all_cols, all_labels),
    ]:
        im = ax.imshow(mat * 100, cmap="Blues", vmin=0, vmax=100)
        ax.set_title(title, fontsize=8)
        ax.set_xticks(np.arange(len(cols)))
        ax.set_xticklabels(cols, rotation=65, ha="right", fontsize=4.8)
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels(rows, fontsize=4.8)
        ax.set_xlabel("Predicted post-hoc class")
        ax.set_ylabel("True class")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.86, pad=0.02)
    cbar.set_label("Row-normalized percentage (%)")
    save(fig, "fig7_mclean_mean_confusion_matrices")


def fig8_fair_reduction() -> None:
    df = ensure_acc_kept(pd.read_csv(FOLLOWUP / "fair_reduction_5seeds.csv"))
    df = df[df["dataset"].isin(["F_new", "V_new", "M_new_drop5_drop7"])]
    reductions = ["raw", "pca", "umap"]
    colors = [PALETTE["gray"], PALETTE["orange"], PALETTE["blue"]]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), sharey=True)
    for ax, ds in zip(axes, ["F_new", "V_new", "M_new_drop5_drop7"]):
        sub = df[df["dataset"] == ds]
        vals = [sub[sub["reduction"] == r]["acc_kept"].mean() * 100 for r in reductions]
        errs = [sub[sub["reduction"] == r]["acc_kept"].std(ddof=0) * 100 for r in reductions]
        ax.bar(np.arange(3), vals, yerr=errs, color=colors, capsize=2)
        ax.set_title(DATASET_LABELS[ds], fontsize=8)
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(["Raw\n2048D", "PCA\n100D", "UMAP\n100D"])
        ax.set_ylim(75, 101)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    axes[0].set_ylabel("acc$_{kept}$ (%)")
    fig.suptitle("Fair reduction comparison under all3 and K=20", fontsize=8)
    save(fig, "fig8_fair_reduction_comparison")


def main() -> None:
    fig1_workflow()
    fig2_main_results()
    fig3_baseline_comparison()
    fig4_risk_coverage()
    fig5_parameter_heatmap()
    fig6_weak_classes()
    fig7_confusion_matrices()
    fig8_fair_reduction()
    print(OUT)


if __name__ == "__main__":
    main()
