#!/usr/bin/env python3
"""Draw final 5-seed confusion matrices directly from per-sample predictions.

This script does not read manuscript text and does not reuse pre-rendered
confusion-matrix images. It recomputes row-normalized matrices from:

    dataset, seed, true_label, aligned_pred_label, predicted_label_for_matrix, kept

Main fixed protocol:
    UMAP dimension = 100
    K = 60
    method = all3 / C3
    seeds = 11, 22, 33, 44, 55

Run from the repository root:
    python code/draw_confusion_from_per_sample.py
"""
from __future__ import annotations

import os
import textwrap
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("PROTOCOLA_RERUN_ROOT", str(REPO_ROOT)))

INPUT_DIR = Path(os.environ.get("PROTOCOLA_PER_SAMPLE_DIR", str(ROOT / "results" / "final_k60" / "confusion_matrices")))
OUT = Path(os.environ.get("PROTOCOLA_DIRECT_CONFUSION_OUT", str(ROOT / "outputs" / "direct_confusion_from_per_sample")))
CSV_OUT = OUT / "csv"
FIG_OUT = OUT / "figures"
for folder in [CSV_OUT, FIG_OUT]:
    folder.mkdir(parents=True, exist_ok=True)

DATASETS = [
    ("F_new", "PV-Fruit", INPUT_DIR / "PV-Fruit" / "per_sample_predictions.csv"),
    ("V_new", "PV-Vegetable", INPUT_DIR / "PV-Vegetable" / "per_sample_predictions.csv"),
    ("M_new", "MCLD-11", INPUT_DIR / "MCLD-11" / "per_sample_predictions.csv"),
    ("G_new", "DFLD-BR-4", INPUT_DIR / "DFLD-BR-4" / "per_sample_predictions.csv"),
]

SEEDS = [11, 22, 33, 44, 55]
UMAP_DIM = 100
N_CLUSTERS = 60
METHOD = "all3"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
    }
)


def clean_label(label: str) -> str:
    text = str(label)
    replacements = {
        "Tomato verticulium wilt": "Tomato Verticillium wilt",
        "Tomato - verticulium wilt": "Tomato - Verticillium wilt",
        "Tomato - Verticulium wilt": "Tomato - Verticillium wilt",
    }
    return replacements.get(text, text)


def wrap_label(label: str, width: int = 14) -> str:
    label = clean_label(label)
    if label.lower() == "rejected":
        return "Rejected"
    parts: list[str] = []
    for chunk in label.replace(" - ", " | ").split(" | "):
        parts.extend(textwrap.wrap(chunk, width=width) or [chunk])
    return "\n".join(parts)


def class_order(df: pd.DataFrame) -> list[str]:
    ordered = (
        df[["true_label_id", "true_label"]]
        .drop_duplicates()
        .sort_values("true_label_id")["true_label"]
        .map(clean_label)
        .tolist()
    )
    return ordered


def normalize_rows(counts: pd.DataFrame) -> pd.DataFrame:
    denom = counts.sum(axis=1).replace(0, np.nan)
    return counts.div(denom, axis=0).fillna(0.0)


def seed_matrix(df_seed: pd.DataFrame, labels: list[str], include_rejected: bool) -> pd.DataFrame:
    cols = labels + (["Rejected"] if include_rejected else [])
    work = df_seed.copy()
    work["true_label"] = work["true_label"].map(clean_label)
    work["predicted_label_for_matrix"] = work["predicted_label_for_matrix"].map(clean_label)
    work["aligned_pred_label"] = work["aligned_pred_label"].map(clean_label)

    if include_rejected:
        pred_col = "predicted_label_for_matrix"
        work.loc[work["kept"].astype(int) == 0, pred_col] = "Rejected"
    else:
        work = work[work["kept"].astype(int) == 1].copy()
        pred_col = "aligned_pred_label"
        work = work[work[pred_col].isin(labels)]

    counts = pd.crosstab(work["true_label"], work[pred_col])
    counts = counts.reindex(index=labels, columns=cols, fill_value=0)
    return normalize_rows(counts)


def average_5seed_matrix(df: pd.DataFrame, labels: list[str], include_rejected: bool) -> pd.DataFrame:
    matrices = []
    for seed in SEEDS:
        sub = df[df["seed"].astype(int) == seed]
        if sub.empty:
            raise RuntimeError(f"Missing seed {seed}")
        matrices.append(seed_matrix(sub, labels, include_rejected))
    values = np.stack([m.to_numpy(dtype=float) for m in matrices], axis=0).mean(axis=0)
    return pd.DataFrame(values, index=matrices[0].index, columns=matrices[0].columns)


def save_matrix_csv(mat: pd.DataFrame, path: Path) -> None:
    out = mat.copy()
    out.insert(0, "true_label", out.index)
    out.to_csv(path, index=False, encoding="utf-8-sig")


def fmt_cell(value_pct: float) -> str:
    if abs(value_pct - round(value_pct)) < 0.05:
        return f"{value_pct:.0f}"
    return f"{value_pct:.1f}"


def draw_matrix(
    ax: plt.Axes,
    mat: pd.DataFrame,
    title: str,
    tick_fontsize: float,
    annot_fontsize: float,
    x_wrap: int,
    y_wrap: int,
    show_axis_labels: bool = True,
):
    vals = mat.to_numpy(dtype=float) * 100
    n_rows, n_cols = vals.shape
    x_edges = np.arange(n_cols + 1)
    y_edges = np.arange(n_rows + 1)
    im = ax.pcolormesh(
        x_edges,
        y_edges,
        vals,
        cmap="Blues",
        vmin=0,
        vmax=100,
        shading="flat",
        edgecolors="white",
        linewidth=0.25,
    )
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=9, pad=12)
    if show_axis_labels:
        ax.set_xlabel("Predicted class", labelpad=10)
        ax.set_ylabel("True class", labelpad=14)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels([wrap_label(c, x_wrap) for c in mat.columns], rotation=48, ha="right", va="top", fontsize=tick_fontsize)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels([wrap_label(r, y_wrap) for r in mat.index], fontsize=tick_fontsize)
    ax.tick_params(axis="x", pad=5)
    ax.tick_params(axis="y", pad=4)
    for i in range(n_rows):
        for j in range(n_cols):
            value = vals[i, j]
            if value < 0.05:
                continue
            ax.text(
                j + 0.5,
                i + 0.5,
                fmt_cell(value),
                ha="center",
                va="center",
                fontsize=annot_fontsize,
                color="white" if value >= 55 else "#2B2B2B",
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def save_figure(fig: plt.Figure, stem: str) -> None:
    for ext, dpi in [("png", 300), ("pdf", None), ("tif", 600)]:
        path = FIG_OUT / f"{stem}.{ext}"
        if dpi:
            fig.savefig(path, dpi=dpi, facecolor="white")
        else:
            fig.savefig(path, facecolor="white")
    plt.close(fig)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = df[
        (df["method"].astype(str) == METHOD)
        & (df["umap_dim"].astype(int) == UMAP_DIM)
        & (df["n_clusters"].astype(int) == N_CLUSTERS)
    ].copy()
    if df.empty:
        raise RuntimeError(f"No rows after filtering method={METHOD}, umap_dim={UMAP_DIM}, n_clusters={N_CLUSTERS}: {path}")
    return df


def main() -> None:
    retained_mats: dict[str, pd.DataFrame] = {}
    all_rejected_mats: dict[str, pd.DataFrame] = {}

    for dataset_id, display_name, path in DATASETS:
        df = load_dataset(path)
        labels = class_order(df)
        retained = average_5seed_matrix(df, labels, include_rejected=False)
        all_plus_rejected = average_5seed_matrix(df, labels, include_rejected=True)
        retained_mats[display_name] = retained
        all_rejected_mats[display_name] = all_plus_rejected
        save_matrix_csv(retained, CSV_OUT / f"{dataset_id}_{display_name}_retained_5seed_avg_from_per_sample.csv")
        save_matrix_csv(all_plus_rejected, CSV_OUT / f"{dataset_id}_{display_name}_all_plus_rejected_5seed_avg_from_per_sample.csv")

    fig = plt.figure(figsize=(23.0, 18.8))
    axes = [
        fig.add_axes([0.035, 0.565, 0.445, 0.365]),
        fig.add_axes([0.505, 0.565, 0.445, 0.365]),
        fig.add_axes([0.035, 0.075, 0.445, 0.365]),
        fig.add_axes([0.505, 0.075, 0.445, 0.365]),
    ]
    settings = {
        "PV-Fruit": dict(tick_fontsize=6.4, annot_fontsize=5.8, x_wrap=13, y_wrap=18),
        "PV-Vegetable": dict(tick_fontsize=5.8, annot_fontsize=5.4, x_wrap=12, y_wrap=17),
        "MCLD-11": dict(tick_fontsize=5.9, annot_fontsize=5.4, x_wrap=12, y_wrap=17),
        "DFLD-BR-4": dict(tick_fontsize=8.0, annot_fontsize=6.5, x_wrap=14, y_wrap=14),
    }
    last_im = None
    for ax, (_, display_name, _) in zip(axes, DATASETS):
        last_im = draw_matrix(
            ax,
            retained_mats[display_name],
            f"{display_name}: retained samples",
            **settings[display_name],
        )
    cax = fig.add_axes([0.945, 0.090, 0.014, 0.825])
    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_label("Row-normalized value (%)")
    fig.suptitle("Four-dataset retained-sample confusion matrices, recomputed from per-sample predictions", fontsize=12, y=0.985)
    save_figure(fig, "direct_Fig10_retained_5seed_avg_from_per_sample")

    for _, display_name, _ in DATASETS:
        mat = retained_mats[display_name]
        n_cols = mat.shape[1]
        n_rows = mat.shape[0]
        fig_w = max(7.0, 0.78 * n_cols + 3.0)
        fig_h = max(6.0, 0.70 * n_rows + 2.7)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = draw_matrix(
            ax,
            mat,
            f"{display_name}: retained samples, 5-seed average",
            tick_fontsize=7.0 if n_rows <= 9 else 6.3,
            annot_fontsize=6.2,
            x_wrap=14,
            y_wrap=18,
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
        cbar.set_label("Row-normalized value (%)")
        fig.subplots_adjust(left=0.24 if n_rows > 9 else 0.18, right=0.91, top=0.88, bottom=0.31)
        save_figure(fig, f"direct_{display_name}_retained_5seed_avg_from_per_sample")

    for _, display_name, _ in DATASETS:
        mat = all_rejected_mats[display_name]
        fig, ax = plt.subplots(figsize=(10.8 if mat.shape[1] > 8 else 7.2, 8.8 if mat.shape[0] > 8 else 6.2))
        im = draw_matrix(
            ax,
            mat,
            f"{display_name}: all samples with Rejected",
            tick_fontsize=5.5 if mat.shape[0] > 8 else 7.0,
            annot_fontsize=5.0,
            x_wrap=11,
            y_wrap=16,
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
        cbar.set_label("Row-normalized value (%)")
        fig.subplots_adjust(left=0.24 if mat.shape[0] > 8 else 0.18, right=0.91, top=0.88, bottom=0.31)
        save_figure(fig, f"direct_{display_name}_all_plus_rejected_5seed_avg_from_per_sample")

    print(f"Input:  {INPUT_DIR}")
    print(f"CSV:    {CSV_OUT}")
    print(f"Figure: {FIG_OUT}")


if __name__ == "__main__":
    main()
