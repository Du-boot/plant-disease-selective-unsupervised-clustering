from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Resolve all input/output paths from the repository root. This works on a
# server, local workstation, or cloned GitHub checkout without path editing.
ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / "figures" / "v13_figure_report" / "source_data" / "confusion_matrices"
OUT = ROOT / "outputs" / "individual_confusion_matrices"
OUT_RETAINED = OUT / "retained_samples"
OUT_ALL_REJECTED = OUT / "all_samples_with_rejected"

for d in [OUT_RETAINED, OUT_ALL_REJECTED]:
    d.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


CONFUSION_FILES = {
    "PV-Fruit": {
        "kept": "PV-Fruit_kept.csv",
        "all_rejected": "PV-Fruit_all_plus_Rejected.csv",
    },
    "PV-Vegetable": {
        "kept": "PV-Vegetable_kept.csv",
        "all_rejected": "PV-Vegetable_all_plus_Rejected.csv",
    },
    "MCLD-11": {
        "kept": "MCLD-11_kept.csv",
        "all_rejected": "MCLD-11_all_plus_Rejected.csv",
    },
    "DFLD-BR-4": {
        "kept": "DFLD-BR-4_kept.csv",
        "all_rejected": "DFLD-BR-4_all_plus_Rejected.csv",
    },
}


# Each output is a standalone manuscript figure, not a cropped panel from Fig. 10.
# The large exterior margins are intentional: long disease names and rotated x labels
# must remain entirely within the exported canvas in Word, PDF, and TIFF.
PANEL_SETTINGS = {
    "PV-Fruit": dict(
        figsize=(11.8, 11.6), tick_fontsize=7.0, annot_fontsize=6.7,
        x_wrap=14, y_wrap=19, left=0.27, right=0.88, bottom=0.29, top=0.90
    ),
    "PV-Vegetable": dict(
        figsize=(12.6, 12.5), tick_fontsize=6.6, annot_fontsize=6.2,
        x_wrap=14, y_wrap=20, left=0.28, right=0.88, bottom=0.30, top=0.90
    ),
    "MCLD-11": dict(
        figsize=(12.6, 12.5), tick_fontsize=6.6, annot_fontsize=6.2,
        x_wrap=14, y_wrap=20, left=0.28, right=0.88, bottom=0.30, top=0.90
    ),
    "DFLD-BR-4": dict(
        figsize=(9.5, 9.3), tick_fontsize=8.0, annot_fontsize=7.2,
        x_wrap=16, y_wrap=18, left=0.25, right=0.87, bottom=0.26, top=0.90
    ),
}


def display_class_name(name: str) -> str:
    replacements = {
        "Tomato verticulium wilt": "Tomato Verticillium wilt",
        "Tomato - verticulium wilt": "Tomato - Verticillium wilt",
        "Tomato - Verticulium wilt": "Tomato - Verticillium wilt",
    }
    return replacements.get(str(name), str(name))


def wrap_label(label: str, width: int) -> str:
    label = display_class_name(label)
    if label.lower() == "rejected":
        return "Rejected"
    parts: list[str] = []
    for chunk in label.replace(" - ", " | ").split(" | "):
        parts.extend(textwrap.wrap(chunk, width=width) or [chunk])
    return "\n".join(parts)


def read_matrix(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    df = pd.read_csv(path)
    row_labels = df.iloc[:, 0].astype(str).tolist()
    col_labels = [str(c) for c in df.columns[1:]]
    mat = df.iloc[:, 1:].astype(float).to_numpy()
    return row_labels, col_labels, mat


def fmt_value(value: float) -> str:
    if abs(value - round(value)) < 0.05:
        return f"{value:.0f}"
    return f"{value:.1f}"


def save_pub(fig: plt.Figure, out_dir: Path, stem: str) -> dict[str, Path]:
    paths = {
        "png": out_dir / f"{stem}.png",
        "pdf": out_dir / f"{stem}.pdf",
        "tif": out_dir / f"{stem}.tif",
    }
    # The figure margins reserve label room; a padded tight export removes only
    # unused canvas around that content. It does not crop labels or cells.
    export_kwargs = {"bbox_inches": "tight", "pad_inches": 0.26, "facecolor": "white"}
    fig.savefig(paths["png"], dpi=300, **export_kwargs)
    fig.savefig(paths["pdf"], **export_kwargs)
    fig.savefig(paths["tif"], dpi=600, **export_kwargs)
    plt.close(fig)
    return paths


def draw_one(csv_path: Path, out_dir: Path, stem: str, title: str, dataset: str) -> dict[str, Path]:
    rows, cols, mat = read_matrix(csv_path)
    settings = PANEL_SETTINGS[dataset]
    fig, ax = plt.subplots(figsize=settings["figsize"], constrained_layout=False)
    im = ax.imshow(mat * 100, cmap="Blues", vmin=0, vmax=100, aspect="equal")
    ax.set_xlim(-0.5, mat.shape[1] - 0.5)
    # Leave a small data-space margin above the first row so tight export never
    # clips the first annotation or the top edge of the matrix.
    ax.set_ylim(mat.shape[0] - 0.5, -0.75)

    ax.set_title(title, fontsize=10, pad=18)
    ax.set_xlabel("Predicted class", labelpad=10, fontsize=8.5)
    ax.set_ylabel("True class", labelpad=10, fontsize=8.5)

    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(
        [wrap_label(c, settings["x_wrap"]) for c in cols],
        rotation=55,
        ha="right",
        va="top",
        fontsize=settings["tick_fontsize"],
    )
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(
        [wrap_label(r, settings["y_wrap"]) for r in rows],
        fontsize=settings["tick_fontsize"],
    )
    ax.tick_params(axis="x", pad=5)
    ax.tick_params(axis="y", pad=3)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat[i, j] * 100
            if value < 0.05:
                continue
            ax.text(
                j,
                i,
                fmt_value(value),
                ha="center",
                va="center",
                fontsize=settings["annot_fontsize"],
                color="white" if value >= 55 else "#2B2B2B",
                clip_on=False,
            )

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Row-normalized value (%)")
    cbar.ax.tick_params(labelsize=7)
    fig.subplots_adjust(
        left=settings["left"],
        right=settings["right"],
        bottom=settings["bottom"],
        top=settings["top"],
    )
    return save_pub(fig, out_dir, stem)


def main() -> None:
    outputs = []
    for dataset, files in CONFUSION_FILES.items():
        kept_csv = CONF / files["kept"]
        all_csv = CONF / files["all_rejected"]
        outputs.append(
            draw_one(
                kept_csv,
                OUT_RETAINED,
                f"{dataset}_retained_5seed_avg_confusion",
                f"{dataset}: retained samples",
                dataset,
            )
        )
        outputs.append(
            draw_one(
                all_csv,
                OUT_ALL_REJECTED,
                f"{dataset}_all_plus_rejected_5seed_avg_confusion",
                f"{dataset}: all samples with Rejected",
                dataset,
            )
        )

    print(f"Input CSV folder: {CONF}")
    print(f"Retained outputs: {OUT_RETAINED}")
    print(f"All+Rejected outputs: {OUT_ALL_REJECTED}")
    for group in [OUT_RETAINED, OUT_ALL_REJECTED]:
        print(f"\n{group}:")
        for p in sorted(group.glob("*")):
            print(f"  {p}")


if __name__ == "__main__":
    main()
