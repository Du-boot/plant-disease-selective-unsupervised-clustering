#!/usr/bin/env python3
"""Create a four-panel confusion-matrix figure from real matrix CSV files.

This script reads the row-normalized confusion-matrix CSV files produced by
protocolA_final_required_patch.py and renders the manuscript-style 2 x 2
confusion-matrix figure.
"""
from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = Path(os.environ.get(
    "PROTOCOLA_CONFUSION_INPUT_DIR",
    str(REPO_ROOT / "outputs" / "protocolA_final_required_patch" / "class_level_5seed"),
))
DEFAULT_OUT_DIR = Path(os.environ.get(
    "PROTOCOLA_CONFUSION_FIGURE_DIR",
    str(REPO_ROOT / "outputs" / "confusion_figures"),
))

DATASETS = [
    ("PV-Fruit", "F_new_5seed_avg_kept_confusion_row_normalized.csv"),
    ("PV-Vegetable", "V_new_5seed_avg_kept_confusion_row_normalized.csv"),
    ("MCLD-11", "M_new_5seed_avg_kept_confusion_row_normalized.csv"),
    ("DFLD-BR-4", "G_new_5seed_avg_kept_confusion_row_normalized.csv"),
]

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.right": False,
        "axes.spines.top": False,
    }
)


def read_matrix(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    df = pd.read_csv(path)
    rows = df.iloc[:, 0].astype(str).tolist()
    cols = [str(c) for c in df.columns[1:]]
    mat = df.iloc[:, 1:].astype(float).to_numpy()
    return rows, cols, mat


def wrap_label(label: str, width: int) -> str:
    label = str(label).replace("Tomato verticulium wilt", "Tomato Verticillium wilt")
    label = label.replace(" - ", "\n")
    lines: list[str] = []
    for part in label.split("\n"):
        lines.extend(textwrap.wrap(part, width=width) or [part])
    return "\n".join(lines)


def fmt_value(v: float) -> str:
    pct = v * 100
    if abs(pct - round(pct)) < 0.05:
        return f"{pct:.0f}"
    return f"{pct:.1f}"


def draw_panel(ax: plt.Axes, title: str, path: Path):
    rows, cols, mat = read_matrix(path)
    vals = mat * 100
    im = ax.imshow(vals, cmap="Blues", vmin=0, vmax=100, aspect="auto", interpolation="nearest")

    ax.set_title(f"{title}: retained samples", fontsize=8, pad=6)
    ax.set_xlabel("Predicted class", fontsize=7, labelpad=5)
    ax.set_ylabel("True class", fontsize=7, labelpad=5)

    n_rows, n_cols = vals.shape
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    x_width = 9 if n_cols >= 11 else 10
    y_width = 12 if n_rows >= 10 else 13
    x_size = 4.3 if n_cols >= 11 else 4.8
    y_size = 4.6 if n_rows >= 10 else 5.0
    if n_cols <= 4:
        x_width, y_width, x_size, y_size = 12, 12, 5.4, 5.4
    ax.set_xticklabels([wrap_label(c, x_width) for c in cols], rotation=58, ha="right", va="top", fontsize=x_size)
    ax.set_yticklabels([wrap_label(r, y_width) for r in rows], fontsize=y_size)
    ax.tick_params(axis="x", pad=2, length=2)
    ax.tick_params(axis="y", pad=2, length=2)

    for i in range(n_rows):
        for j in range(n_cols):
            v = mat[i, j]
            if v < 0.0005:
                continue
            color = "white" if v >= 0.55 else "#30343B"
            ax.text(j, i, fmt_value(v), ha="center", va="center", fontsize=4.6, color=color)

    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.2))
    ims = []
    for ax, (title, filename) in zip(axes.ravel(), DATASETS):
        path = args.input_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        ims.append(draw_panel(ax, title, path))

    cax = fig.add_axes([0.925, 0.155, 0.022, 0.69])
    cbar = fig.colorbar(ims[-1], cax=cax)
    cbar.set_label("Row-normalized value (%)", fontsize=7)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle("Four-dataset retained-sample confusion matrices", fontsize=9, y=0.985)
    fig.subplots_adjust(left=0.075, right=0.90, top=0.94, bottom=0.105, wspace=0.22, hspace=0.39)

    stem = args.out / "Fig10_four_dataset_kept_confusion_matrices"
    fig.savefig(f"{stem}.png", dpi=450, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(f"{stem}.tif", dpi=600, bbox_inches="tight", pad_inches=0.04)
    print(stem)


if __name__ == "__main__":
    main()
