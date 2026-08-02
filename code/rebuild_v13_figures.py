#!/usr/bin/env python3
"""Rebuild V13 manuscript figures from archived CSV files and representative images.

This script draws figures only from traceable local files under:
  source_data/

It does not rerun feature extraction, UMAP, clustering, or any AI generation.
Those full experiment scripts are archived under code/.
"""
from __future__ import annotations

import csv
import html
import json
import os
import re
import shutil
import textwrap
import copy
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
RERUN_ROOT = Path(os.environ.get("PROTOCOLA_RERUN_ROOT", str(REPO_ROOT)))
SERVER_DATA_ROOT = Path(os.environ.get("PROTOCOLA_DATA_ROOT", str(REPO_ROOT / "data" / "images_clean")))
ROOT = Path(os.environ.get("PROTOCOLA_FIGURE_REPORT_ROOT", str(RERUN_ROOT / "figures" / "v13_figure_report")))

DATA = ROOT / "source_data"
MAIN = DATA / "main_results"
DINO = DATA / "dinov2"
CONF = DATA / "confusion_matrices"
RAW_IMAGES = DATA / "representative_images"
OUT = ROOT / "reproduced_figures"
MAIN_FIG = OUT / "Main_Figures"
SUPP_FIG = OUT / "Supplementary_Figures"
REPORT_ASSETS = OUT / "report_assets"

for d in [MAIN_FIG, SUPP_FIG, REPORT_ASSETS]:
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

PALETTE = {
    "blue": "#31688E",
    "teal": "#35A77C",
    "orange": "#F28E2B",
    "red": "#D95F5F",
    "purple": "#7A68A6",
    "gray": "#6B7280",
    "light": "#E8EDF3",
}

DATASET_LABELS = {
    "F_new": "PV-Fruit",
    "V_new": "PV-Vegetable",
    "M_new": "MCLD-11",
    "G_new": "DFLD-BR-4",
}

MAIN4_ORDER = ["F_new", "V_new", "M_new", "G_new"]
DINO4_ORDER = ["F_new", "V_new", "M_new", "G_new"]

REPRESENTATIVE_LABELS = {
    "PV-Fruit": {
        0: "Apple - apple scab",
        1: "Apple - black rot",
        2: "Apple - cedar apple rust",
        3: "Cherry - powdery mildew",
        4: "Grape - esca black measles",
        5: "Grape - leaf blight",
        6: "Orange - citrus greening",
        7: "Peach - bacterial spot",
        8: "Strawberry - leaf scorch",
    },
    "PV-Vegetable": {
        0: "Corn - common rust",
        1: "Corn - northern leaf blight",
        2: "Bell pepper - bacterial spot",
        3: "Potato - early blight",
        4: "Potato - late blight",
        5: "Squash - powdery mildew",
        6: "Tomato - bacterial spot",
        7: "Tomato - late blight",
        8: "Tomato - septoria leaf spot",
        9: "Tomato - two-spotted spider mite",
        10: "Tomato - mosaic virus",
        11: "Tomato - yellow leaf curl virus",
    },
    "MCLD-11": {
        0: "Cashew - leaf miner",
        1: "Cashew - red rust",
        2: "Corn - leaf blight",
        3: "Corn - streak virus",
        4: "Potato - fungi",
        5: "Potato - nematode",
        6: "Rice - bacterial leaf blight",
        7: "Rice - brown spot",
        8: "Rice - leaf blast",
        9: "Tomato - septoria leaf spot",
        10: "Tomato - Verticillium wilt",
    },
    "DFLD-BR-4": {
        0: "Gourd",
        1: "Hibiscus",
        2: "Papaya",
        3: "Zucchini",
    },
}

SERVER_CONFUSION_ALIASES = {
    "F_new_5seed_avg_kept_confusion_row_normalized.csv": "PV-Fruit_kept.csv",
    "F_new_5seed_avg_all_confusion_with_rejected_row_normalized.csv": "PV-Fruit_all_plus_Rejected.csv",
    "V_new_5seed_avg_kept_confusion_row_normalized.csv": "PV-Vegetable_kept.csv",
    "V_new_5seed_avg_all_confusion_with_rejected_row_normalized.csv": "PV-Vegetable_all_plus_Rejected.csv",
    "M_new_5seed_avg_kept_confusion_row_normalized.csv": "MCLD-11_kept.csv",
    "M_new_5seed_avg_all_confusion_with_rejected_row_normalized.csv": "MCLD-11_all_plus_Rejected.csv",
    "G_new_5seed_avg_kept_confusion_row_normalized.csv": "DFLD-BR-4_kept.csv",
    "G_new_5seed_avg_all_confusion_with_rejected_row_normalized.csv": "DFLD-BR-4_all_plus_Rejected.csv",
}

REPRESENTATIVE_DATASETS = {
    "PV-Fruit": "F_new",
    "PV-Vegetable": "V_new",
    "MCLD-11": "M_new",
    "DFLD-BR-4": "G_new",
}


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists() or src.is_dir():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def server_image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def prepare_representative_images_from_server() -> None:
    for display_name, dataset in REPRESENTATIVE_DATASETS.items():
        out_dir = RAW_IMAGES / display_name
        if out_dir.exists() and any(out_dir.iterdir()):
            continue
        files = server_image_files(SERVER_DATA_ROOT / dataset)
        if not files:
            continue
        by_class: dict[str, Path] = {}
        for p in files:
            class_id = p.name.split("_", 1)[0]
            by_class.setdefault(class_id, p)
        selected = [by_class[k] for k in sorted(by_class, key=lambda x: int(x) if str(x).isdigit() else str(x))[:5]]
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in selected:
            shutil.copy2(p, out_dir / p.name)


def prepare_source_data_from_server() -> None:
    """Collect newly rerun server CSVs into the compact source_data layout."""
    if not RERUN_ROOT.exists():
        return

    MAIN.mkdir(parents=True, exist_ok=True)
    CONF.mkdir(parents=True, exist_ok=True)
    DINO.mkdir(parents=True, exist_ok=True)
    RAW_IMAGES.mkdir(parents=True, exist_ok=True)

    followup = RERUN_ROOT / "outputs" / "protocolA_followup" / "followup_results"
    required = RERUN_ROOT / "outputs" / "protocolA_final_required_patch"
    supplements = RERUN_ROOT / "outputs" / "protocolA_final_supplements"

    for name in [
        "clean_main.csv",
        "random_same_coverage_completed.csv",
        "fair_reduction_5seeds.csv",
        "parameter_sensitivity_5seeds.csv",
        "parameter_trend_smallK.csv",
        "internal_metrics.csv",
        "cross_seed_stability.csv",
    ]:
        copy_if_exists(followup / name, MAIN / name)

    for src in [
        required / "bootstrap_seedlevel" / "method_difference_seedlevel_bootstrap_ci.csv",
        required / "bootstrap_seedlevel" / "summary_seedlevel_bootstrap_ci.csv",
        supplements / "bootstrap_seedlevel" / "method_difference_seedlevel_bootstrap_ci.csv",
    ]:
        if copy_if_exists(src, MAIN / "summary_seedlevel_bootstrap_ci.csv"):
            break

    for src in [
        required / "risk_coverage_discrete" / "risk_coverage_discrete_operating_points.csv",
        supplements / "risk_coverage_discrete" / "risk_coverage_discrete_operating_points.csv",
    ]:
        if copy_if_exists(src, MAIN / "risk_coverage_discrete_operating_points.csv"):
            break

    for class_root in [required / "class_level_5seed", supplements / "class_level_5seed"]:
        if not class_root.exists():
            continue
        for src in class_root.glob("*.csv"):
            copy_if_exists(src, CONF / src.name)
            if src.name in SERVER_CONFUSION_ALIASES:
                copy_if_exists(src, CONF / SERVER_CONFUSION_ALIASES[src.name])

    copy_if_exists(supplements / "m_new_full" / "M_new_full_main.csv", CONF / "M_new_full_main.csv")
    copy_if_exists(supplements / "m_new_full" / "M_new_exact_duplicate_groups.csv", CONF / "M_new_exact_duplicate_groups.csv")

    dinov2_candidates = [
        RERUN_ROOT / "outputs" / "external_baselines_dinov2" / "web_report_external_baselines_summary" / "summary_convnext_vs_dinov2_key.csv",
        RERUN_ROOT / "outputs" / "external_baselines_dinov2" / "summary_convnext_vs_dinov2_key.csv",
        REPO_ROOT / "external_baselines" / "dinov2" / "summary_convnext_vs_dinov2_key.csv",
    ]
    for src in dinov2_candidates:
        if copy_if_exists(src, DINO / "summary_convnext_vs_dinov2_key.csv"):
            break

    dinov2_ci_candidates = [
        RERUN_ROOT / "outputs" / "external_baselines_dinov2" / "bootstrap_seedlevel_dinov2" / "dinov2_seedlevel_paired_bootstrap_ci.csv",
        REPO_ROOT / "external_baselines" / "dinov2" / "bootstrap_seedlevel_dinov2" / "dinov2_seedlevel_paired_bootstrap_ci.csv",
    ]
    for src in dinov2_ci_candidates:
        if copy_if_exists(src, DINO / "dinov2_seedlevel_paired_bootstrap_ci.csv"):
            break

    prepare_representative_images_from_server()


def save_figure(fig: plt.Figure, folder: Path, stem: str, pad_inches: float = 0.04) -> dict[str, str]:
    paths: dict[str, str] = {}
    for ext in ["png", "pdf", "tif"]:
        path = folder / f"{stem}.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=pad_inches)
        elif ext == "tif":
            fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=pad_inches)
        else:
            fig.savefig(path, bbox_inches="tight", pad_inches=pad_inches)
        paths[ext] = path.relative_to(ROOT).as_posix()
    plt.close(fig)
    return paths


def pct(x: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return np.asarray(x, dtype=float) * 100


def mean_sd(df: pd.DataFrame, group_cols: list[str], value: str) -> pd.DataFrame:
    return df.groupby(group_cols)[value].agg(["mean", "std"]).reset_index()


def first_number(value) -> float:
    """Parse the mean from cells such as '0.389 +/- 0.004' or '0.389 卤 0.004'."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value)
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return np.nan
    return float(match.group(0))


def read_matrix(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    df = pd.read_csv(path)
    row_labels = df.iloc[:, 0].astype(str).tolist()
    col_labels = [str(c) for c in df.columns[1:]]
    mat = df.iloc[:, 1:].astype(float).to_numpy()
    return row_labels, col_labels, mat


def wrap_label(label: str, width: int = 18) -> str:
    label = str(label)
    if label.lower() == "rejected":
        return "Rejected"
    parts: list[str] = []
    for chunk in label.replace(" - ", " | ").split(" | "):
        parts.extend(textwrap.wrap(chunk, width=width) or [chunk])
    return "\n".join(parts)


def display_class_name(name: str) -> str:
    name = str(name)
    replacements = {
        "Tomato verticulium wilt": "Tomato Verticillium wilt",
        "Tomato - verticulium wilt": "Tomato - Verticillium wilt",
        "Tomato - Verticulium wilt": "Tomato - Verticillium wilt",
    }
    return replacements.get(name, name)


def class_id_from_filename(path: Path) -> int | None:
    match = re.match(r"^(\d+)", path.name)
    return int(match.group(1)) if match else None


def representative_title(dataset: str, path: Path) -> str:
    class_id = class_id_from_filename(path)
    if class_id is not None and class_id in REPRESENTATIVE_LABELS.get(dataset, {}):
        return REPRESENTATIVE_LABELS[dataset][class_id]
    stem = path.stem.split("__")[0].replace("_", " ")
    return stem


def fmt_pct_cell(x: float) -> str:
    v = x * 100
    return f"{v:.0f}" if abs(v - round(v)) < 0.05 else f"{v:.1f}"


def fig01_framework() -> dict[str, str]:
    labels = [
        "Unlabeled\nleaf images",
        "Frozen\nConvNeXt/DINOv2",
        "UMAP\n100-D",
        "KMeans / Birch /\nAgglomerative\nK=60 main",
        "Hungarian\ncluster alignment",
        "C2 / C3\nconsensus",
        "Reject\nuncertain samples",
        "Post-hoc\nexternal evaluation",
    ]
    fig, ax = plt.subplots(figsize=(11.2, 2.1))
    ax.set_axis_off()
    x0, y, w, h, gap = 0.02, 0.40, 0.105, 0.34, 0.018
    for i, lab in enumerate(labels):
        x = x0 + i * (w + gap)
        face = "#F8FAFC"
        edge = "#64748B"
        if i in {5, 6}:
            face = "#EDF8F4"
            edge = "#1B7F6B"
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.025",
            linewidth=1,
            facecolor=face,
            edgecolor=edge,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, lab, ha="center", va="center", fontsize=8)
        if i < len(labels) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + w + 0.004, y + h / 2),
                    (x + w + gap - 0.004, y + h / 2),
                    arrowstyle="-|>",
                    mutation_scale=10,
                    linewidth=0.8,
                    color="#334155",
                )
            )
    ax.text(
        0.5,
        0.17,
        "Disease labels are not used for feature extraction, UMAP, clustering, alignment, or rejection. Main experiments use K=60.",
        ha="center",
        va="center",
        fontsize=8,
        color="#475569",
    )
    return save_figure(fig, MAIN_FIG, "Fig01_Detailed_Framework")


def fig02_representative_images() -> dict[str, str]:
    datasets = ["PV-Fruit", "PV-Vegetable", "MCLD-11", "DFLD-BR-4"]
    fig, axes = plt.subplots(len(datasets), 5, figsize=(11.0, 8.3))
    for r, ds in enumerate(datasets):
        imgs = sorted((RAW_IMAGES / ds).glob("*"))
        for c in range(5):
            ax = axes[r, c]
            ax.set_axis_off()
            if c < len(imgs):
                img = Image.open(imgs[c]).convert("RGB")
                ax.imshow(img)
                title = representative_title(ds, imgs[c])
                ax.set_title(wrap_label(title, 20), fontsize=7, pad=5)
            if c == 0:
                ax.text(-0.08, 0.5, ds, transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=8)
    fig.tight_layout(h_pad=1.9, w_pad=1.0)
    return save_figure(fig, MAIN_FIG, "Fig02_Representative_Dataset_Images")


def fig03_mapping_schematic() -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    ax.set_axis_off()
    left = [(0.08, 0.72, "Cluster 2"), (0.08, 0.50, "Cluster 7"), (0.08, 0.28, "Cluster 14")]
    right = [(0.68, 0.72, "Apple black rot"), (0.68, 0.50, "Tomato late blight")]
    for x, y, text in left:
        ax.add_patch(FancyBboxPatch((x, y), 0.18, 0.12, boxstyle="round,pad=0.02", fc="#EFF6FF", ec="#31688E"))
        ax.text(x + 0.09, y + 0.06, text, ha="center", va="center", fontsize=9)
    for x, y, text in right:
        ax.add_patch(FancyBboxPatch((x, y), 0.22, 0.12, boxstyle="round,pad=0.02", fc="#F0FDF4", ec="#35A77C"))
        ax.text(x + 0.11, y + 0.06, text, ha="center", va="center", fontsize=9)
    arrows = [((0.29, 0.78), (0.635, 0.78)), ((0.29, 0.56), (0.635, 0.78)), ((0.29, 0.34), (0.635, 0.56))]
    for a, b in arrows:
        ax.add_patch(
            FancyArrowPatch(
                a,
                b,
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.1,
                color="#334155",
                shrinkA=4,
                shrinkB=18,
            )
        )
    ax.text(0.5, 0.12, "Many-to-one post-hoc mapping is applied only after clustering and rejection are fixed.", ha="center", fontsize=9)
    ax.set_title("Post-hoc many-to-one semantic mapping", fontsize=11)
    return save_figure(fig, MAIN_FIG, "Fig03_PostHoc_ManyToOne_Mapping")


def fig04_main_results() -> dict[str, str]:
    df = pd.read_csv(MAIN / "clean_main.csv")
    order = MAIN4_ORDER
    df = df[
        (df["dataset"].isin(order))
        & (df["method"] == "all3")
        & (df["reduction"] == "umap")
        & (df["dim"] == 100)
        & (df["k"] == 60)
    ]
    rows = []
    for ds in order:
        sub = df[df["dataset"] == ds]
        if sub.empty:
            continue
        rows.append(
            {
                "dataset": DATASET_LABELS[ds],
                "Acc_kept": sub["acc_kept"].mean() * 100,
                "Acc_kept_sd": sub["acc_kept"].std(ddof=1) * 100,
                "Coverage": (1 - sub["rejection_rate"]).mean() * 100,
                "Coverage_sd": sub["rejection_rate"].std(ddof=1) * 100,
                "Conservative": sub["overall_accuracy"].mean() * 100,
                "Conservative_sd": sub["overall_accuracy"].std(ddof=1) * 100,
            }
        )
    d = pd.DataFrame(rows)
    metrics = [("Acc_kept", PALETTE["blue"]), ("Coverage", PALETTE["teal"]), ("Conservative", PALETTE["orange"])]
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    width = 0.23
    for i, (metric, color) in enumerate(metrics):
        bars = ax.bar(
            x + (i - 1) * width,
            d[metric],
            width,
            yerr=d[f"{metric}_sd"],
            capsize=2,
            color=color,
            label=metric,
        )
        for bar, value, err in zip(bars, d[metric], d[f"{metric}_sd"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + float(err) + 1.4,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="#334155",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(d["dataset"])
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 112)
    ax.set_title("Main C3 results under UMAP100 + K60")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.25, top=0.88)
    return save_figure(fig, MAIN_FIG, "Fig04_Main_C3_Results")


def fig05_all_vs_retained_metrics() -> dict[str, str]:
    summary_path = MAIN / "summary_all_vs_kept_cluster_metrics.csv"
    if summary_path.exists():
        d = pd.read_csv(summary_path)
        d = d[d["dataset"].isin(MAIN4_ORDER)]
        for col in d.columns:
            if col != "dataset":
                d[col] = d[col].map(first_number)
    else:
        raw = pd.read_csv(MAIN / "clean_main.csv")
        raw = raw[
            (raw["dataset"].isin(MAIN4_ORDER))
            & (raw["method"] == "all3")
            & (raw["reduction"] == "umap")
            & (raw["dim"] == 100)
            & (raw["k"] == 60)
        ]
        d = (
            raw.groupby("dataset")[["ari_all", "ari_kept", "nmi_all", "nmi_kept", "ami_all", "ami_kept"]]
            .mean()
            .reset_index()
        )
        d = d.rename(
            columns={
                "ari_all": "ARI_all",
                "ari_kept": "ARI_kept",
                "nmi_all": "NMI_all",
                "nmi_kept": "NMI_kept",
                "ami_all": "AMI_all",
                "ami_kept": "AMI_kept",
            }
        )
    d["order"] = d["dataset"].map({ds: i for i, ds in enumerate(MAIN4_ORDER)})
    d = d.sort_values("order")
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.4), sharey=True)
    y = np.arange(len(d))
    for ax, metric in zip(axes, ["ARI", "NMI", "AMI"]):
        all_col = f"{metric}_all"
        kept_col = f"{metric}_kept"
        ax.hlines(y, d[all_col], d[kept_col], color="#CBD5E1", linewidth=1.2)
        ax.scatter(d[all_col], y, s=28, color="#64748B", label="All samples", zorder=3)
        ax.scatter(d[kept_col], y, s=28, color=PALETTE["teal"], label="C3 retained", zorder=3)
        for yi, all_v, kept_v in zip(y, d[all_col], d[kept_col]):
            delta = kept_v - all_v
            ax.text(
                1.045,
                yi,
                f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}",
                ha="right",
                va="center",
                fontsize=6,
                color=PALETTE["teal"] if delta >= 0 else PALETTE["red"],
            )
        ax.set_yticks(y)
        ax.set_yticklabels([DATASET_LABELS[x] for x in d["dataset"]], fontsize=7)
        ax.set_title(metric)
        ax.set_xlim(0.00, 1.08)
        ax.grid(axis="x", color="#E5E7EB", linewidth=0.6)
        ax.invert_yaxis()
    axes[0].set_xlabel("External clustering metric")
    axes[1].set_xlabel("External clustering metric")
    axes[2].set_xlabel("External clustering metric")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=2, fontsize=7)
    fig.suptitle("All samples vs C3-retained samples", fontsize=10, y=0.98)
    fig.subplots_adjust(left=0.19, right=0.985, top=0.83, bottom=0.23, wspace=0.26)
    return save_figure(fig, MAIN_FIG, "Fig05_All_vs_Retained_Clustering_Metrics")


def fig06_fair_reduction() -> dict[str, str]:
    summary_path = MAIN / "summary_fair_reduction.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
    else:
        raw = pd.read_csv(MAIN / "fair_reduction_5seeds.csv")
        raw = raw[(raw["dataset"].isin(MAIN4_ORDER)) & (raw["method"] == "all3")]
        df = (
            raw.groupby(["dataset", "reduction"])
            .agg(
                acc_kept_mean=("acc_kept", "mean"),
                acc_kept_std=("acc_kept", "std"),
                rejection_rate_mean=("rejection_rate", "mean"),
                rejection_rate_std=("rejection_rate", "std"),
            )
            .reset_index()
        )
    df = df[df["dataset"].isin(MAIN4_ORDER)]
    reductions = ["raw", "pca", "umap"]
    colors = {"raw": PALETTE["gray"], "pca": PALETTE["orange"], "umap": PALETTE["blue"]}
    labels = {"raw": "Raw 2048D", "pca": "PCA 100D", "umap": "UMAP 100D"}
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    x = np.arange(len(MAIN4_ORDER))
    width = 0.23
    offsets = {"raw": -width, "pca": 0.0, "umap": width}
    for r in reductions:
        sub = df[df["reduction"] == r].set_index("dataset").reindex(MAIN4_ORDER)
        vals = sub["acc_kept_mean"].astype(float).to_numpy() * 100
        errs = sub["acc_kept_std"].astype(float).fillna(0).to_numpy() * 100
        bars = ax.bar(
            x + offsets[r],
            vals,
            width,
            yerr=errs,
            capsize=3,
            color=colors[r],
            label=labels[r],
        )
        for bar, value, err in zip(bars, vals, errs):
            if not np.isfinite(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + float(err) + 0.7,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="#334155",
            )
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[ds] for ds in MAIN4_ORDER])
    ax.set_ylabel("Acc_kept (%)")
    ax.set_ylim(80, 101.5)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.14), fontsize=7)
    ax.set_title("Fair Raw/PCA/UMAP comparison")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.24)
    return save_figure(fig, MAIN_FIG, "Fig06_Fair_Dimensionality_Reduction")


def fig07_parameter_sensitivity() -> dict[str, str]:
    df = pd.read_csv(MAIN / "parameter_sensitivity_5seeds.csv")
    df = df[(df["method"] == "all3") & (df["dataset"].isin(MAIN4_ORDER))]
    df = df[(df["dim"] == 100) & (df["k"].isin([20, 60]))].copy()
    df["coverage"] = 1 - df["rejection_rate"]
    grouped = (
        df.groupby(["dataset", "k"])
        .agg(
            acc_mean=("acc_kept", "mean"),
            acc_sd=("acc_kept", "std"),
            cov_mean=("coverage", "mean"),
            cov_sd=("coverage", "std"),
        )
        .reset_index()
    )
    x = np.arange(len(MAIN4_ORDER))
    width = 0.34
    colors = {20: "#94A3B8", 60: PALETTE["teal"]}
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharex=True)
    for ax, metric, ylabel in [
        (axes[0], "acc", "Acc_kept (%)"),
        (axes[1], "cov", "Coverage (%)"),
    ]:
        for offset, k in [(-width / 2, 20), (width / 2, 60)]:
            vals = []
            errs = []
            for ds in MAIN4_ORDER:
                row = grouped[(grouped["dataset"] == ds) & (grouped["k"] == k)]
                vals.append(float(row[f"{metric}_mean"].iloc[0]) * 100 if not row.empty else np.nan)
                errs.append(float(row[f"{metric}_sd"].iloc[0]) * 100 if not row.empty else 0.0)
            bars = ax.bar(
                x + offset,
                vals,
                width,
                yerr=errs,
                capsize=3,
                label=f"K={k}" + (" final" if k == 60 else ""),
                color=colors[k],
                edgecolor="white",
                linewidth=0.6,
            )
            for bar, val in zip(bars, vals):
                if np.isfinite(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        val + 1.2,
                        f"{val:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=6.5,
                        color="#334155",
                    )
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS[ds] for ds in MAIN4_ORDER], rotation=0)
        ax.set_ylim(0, 112)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.6)
        ax.set_title(ylabel.split(" ")[0])
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.18), ncol=2)
    fig.suptitle("Parameter sensitivity at UMAP100: K=20 reference vs K=60 final configuration", y=0.97)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.22, wspace=0.22)
    return save_figure(fig, MAIN_FIG, "Fig07_Parameter_Sensitivity")


def fig08_dinov2_baseline() -> dict[str, str]:
    path = DINO / "summary_convnext_vs_dinov2_key.csv"
    if not path.exists():
        fig, ax = plt.subplots(figsize=(5.8, 3.2))
        ax.set_axis_off()
        ax.text(
            0.5,
            0.58,
            "DINOv2 baseline summary was not found",
            ha="center",
            va="center",
            fontsize=11,
            weight="bold",
        )
        ax.text(
            0.5,
            0.40,
            "Run external_dinov2_baselines.py or copy\nsummary_convnext_vs_dinov2_key.csv into source_data/dinov2/.",
            ha="center",
            va="center",
            fontsize=8,
            color="#475569",
        )
        return save_figure(fig, MAIN_FIG, "Fig08_External_DINOv2_Baseline")
    df = pd.read_csv(path)
    keep_methods = ["ConvNeXt+UMAP+all3", "DINOv2+KMeans", "DINOv2+UMAP+all3"]
    df = df[(df["dataset"].isin(DINO4_ORDER)) & (df["method"].isin(keep_methods))]
    markers = {"ConvNeXt+UMAP+all3": "o", "DINOv2+KMeans": "s", "DINOv2+UMAP+all3": "D"}
    method_colors = {"ConvNeXt+UMAP+all3": PALETTE["blue"], "DINOv2+KMeans": PALETTE["orange"], "DINOv2+UMAP+all3": PALETTE["teal"]}
    dataset_colors = {
        "F_new": PALETTE["gray"],
        "V_new": PALETTE["orange"],
        "M_new": PALETTE["blue"],
        "G_new": PALETTE["teal"],
    }
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    for ds in DINO4_ORDER:
        sub = df[df["dataset"] == ds]
        for _, r in sub.iterrows():
            method = r["method"]
            x = r["coverage_mean"] * 100
            y = r["acc_kept_mean"] * 100
            ax.errorbar(
                x,
                y,
                xerr=r["coverage_sd"] * 100,
                yerr=r["acc_kept_sd"] * 100,
                fmt=markers[method],
                color=dataset_colors[ds],
                mec=method_colors[method],
                mew=1.1,
                ms=7.0,
                capsize=2,
                alpha=0.95,
                linestyle="none",
            )
    ax.set_xlim(64, 102)
    ax.set_ylim(68, 101.5)
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Acc_kept (%)")
    ax.grid(color="#E5E7EB", linewidth=0.6)
    dataset_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=7, markerfacecolor=dataset_colors[ds], markeredgecolor="white", label=DATASET_LABELS[ds])
        for ds in DINO4_ORDER
    ]
    method_handles = [
        Line2D([0], [0], marker=markers[m], linestyle="None", markersize=7, markerfacecolor="white", markeredgecolor=method_colors[m], markeredgewidth=1.2, label=labels)
        for m, labels in [
            ("ConvNeXt+UMAP+all3", "ConvNeXt + UMAP + C3"),
            ("DINOv2+KMeans", "DINOv2 + KMeans"),
            ("DINOv2+UMAP+all3", "DINOv2 + UMAP + C3"),
        ]
    ]
    leg1 = ax.legend(handles=dataset_handles, title="Dataset", loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, title_fontsize=7)
    ax.add_artist(leg1)
    ax.legend(handles=method_handles, title="Method", loc="lower left", bbox_to_anchor=(1.01, 0.0), fontsize=7, title_fontsize=7)
    ax.set_title("External DINOv2 baseline and cross-backbone validation")
    fig.subplots_adjust(left=0.09, right=0.75, top=0.90, bottom=0.12)
    return save_figure(fig, MAIN_FIG, "Fig08_External_DINOv2_Baseline")


def fig09_weak_classes() -> dict[str, str]:
    df = pd.read_csv(CONF / "weak_or_high_rejection_classes_5seed.csv")
    df = df[df["dataset"].isin(["V_new", "M_new"])].copy()
    # Use the most relevant classes: low Acc_kept or high rejection.
    df["score"] = (1 - df["acc_kept_mean"]) + df["rejection_rate_mean"]
    df = df.sort_values("score", ascending=False).head(10).iloc[::-1]
    labels = [wrap_label(f"{DATASET_LABELS[d]}: {display_class_name(c)}", 31) for d, c in zip(df["dataset"], df["class_name"])]
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    ax.barh(y - 0.18, df["acc_kept_mean"] * 100, height=0.34, color=PALETTE["blue"], label="Acc_kept")
    ax.barh(y + 0.18, df["rejection_rate_mean"] * 100, height=0.34, color=PALETTE["red"], label="Rejection")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Percentage (%)")
    ax.set_xlim(0, 105)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.6)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7)
    ax.set_title("Class-level weak and high-rejection categories")
    fig.subplots_adjust(left=0.34, right=0.82, top=0.90, bottom=0.13)
    return save_figure(fig, MAIN_FIG, "Fig09_Weak_and_HighRejection_Classes")


CONFUSION_MAP = {
    "PV-Fruit": ("PV-Fruit_kept.csv", "PV-Fruit_all_plus_Rejected.csv"),
    "PV-Vegetable": ("PV-Vegetable_kept.csv", "PV-Vegetable_all_plus_Rejected.csv"),
    "MCLD-11": ("MCLD-11_kept.csv", "MCLD-11_all_plus_Rejected.csv"),
    "DFLD-BR-4": ("DFLD-BR-4_kept.csv", "DFLD-BR-4_all_plus_Rejected.csv"),
}


def draw_confusion_panel(
    ax: plt.Axes,
    csv_path: Path,
    title: str,
    show_labels: bool = False,
    tick_fontsize: float = 5.4,
    annot_fontsize: float = 5.2,
    x_wrap: int = 13,
    y_wrap: int = 17,
):
    rows, cols, mat = read_matrix(csv_path)
    rows = [display_class_name(r) for r in rows]
    cols = [display_class_name(c) for c in cols]
    im = ax.imshow(mat * 100, cmap="Blues", vmin=0, vmax=100, aspect="equal")
    ax.set_xlim(-0.5, mat.shape[1] - 0.5)
    # Leave a small data-space margin above the first row so tight export never
    # clips the first annotation or the top edge of the matrix.
    ax.set_ylim(mat.shape[0] - 0.5, -0.75)
    ax.set_title(title, fontsize=10, pad=18)
    ax.set_xlabel("Predicted class", labelpad=10, fontsize=8.5)
    ax.set_ylabel("True class", labelpad=10, fontsize=8.5)
    if show_labels:
        ax.set_xticks(np.arange(len(cols)))
        ax.set_xticklabels([wrap_label(c, x_wrap) for c in cols], rotation=55, ha="right", va="top", fontsize=tick_fontsize)
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels([wrap_label(r, y_wrap) for r in rows], fontsize=tick_fontsize)
        ax.tick_params(axis="x", pad=5)
        ax.tick_params(axis="y", pad=3)
    else:
        xlabels = [str(i) for i in range(len(cols))]
        if cols and str(cols[-1]).lower() == "rejected":
            xlabels[-1] = "R"
        ax.set_xticks(np.arange(len(cols)))
        ax.set_xticklabels(xlabels, fontsize=6)
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels([str(i) for i in range(len(rows))], fontsize=6)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j] * 100
            if val < 0.05:
                continue
            text = f"{val:.0f}" if abs(val - round(val)) < 0.05 else f"{val:.1f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=annot_fontsize, color="white" if val >= 55 else "#2B2B2B", clip_on=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


CONFUSION_STANDALONE_SETTINGS = {
    "PV-Fruit": dict(figsize=(11.8, 11.6), tick_fontsize=7.0, annot_fontsize=6.7, x_wrap=14, y_wrap=19, left=0.27, right=0.88, bottom=0.29, top=0.90),
    "PV-Vegetable": dict(figsize=(12.6, 12.5), tick_fontsize=6.6, annot_fontsize=6.2, x_wrap=14, y_wrap=20, left=0.28, right=0.88, bottom=0.30, top=0.90),
    "MCLD-11": dict(figsize=(12.6, 12.5), tick_fontsize=6.6, annot_fontsize=6.2, x_wrap=14, y_wrap=20, left=0.28, right=0.88, bottom=0.30, top=0.90),
    "DFLD-BR-4": dict(figsize=(9.5, 9.3), tick_fontsize=8.0, annot_fontsize=7.2, x_wrap=16, y_wrap=18, left=0.25, right=0.87, bottom=0.26, top=0.90),
}


def standalone_confusion_figure(csv_path: Path, dataset: str, title: str, folder: Path, stem: str) -> dict[str, str]:
    settings = CONFUSION_STANDALONE_SETTINGS[dataset]
    fig, ax = plt.subplots(figsize=settings["figsize"])
    im = draw_confusion_panel(
        ax,
        csv_path,
        title,
        show_labels=True,
        tick_fontsize=settings["tick_fontsize"],
        annot_fontsize=settings["annot_fontsize"],
        x_wrap=settings["x_wrap"],
        y_wrap=settings["y_wrap"],
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Row-normalized value (%)")
    cbar.ax.tick_params(labelsize=7)
    fig.subplots_adjust(
        left=settings["left"],
        right=settings["right"],
        bottom=settings["bottom"],
        top=settings["top"],
    )
    return save_figure(fig, folder, stem, pad_inches=0.26)


def fig10_kept_confusions() -> dict[str, dict[str, str]]:
    outputs = {}
    for letter, dataset in zip("abcd", CONFUSION_MAP):
        stem = f"Fig10{letter}_{dataset.replace('-', '_')}_Kept_Confusion_Matrix"
        outputs[f"Fig10{letter}"] = standalone_confusion_figure(
            CONF / CONFUSION_MAP[dataset][0],
            dataset,
            f"{dataset}: retained samples",
            MAIN_FIG,
            stem,
        )
    return outputs


def supplementary_confusions() -> dict[str, dict[str, str]]:
    outputs = {}
    for letter, dataset in zip("abcd", CONFUSION_MAP):
        stem = f"FigS01{letter}_{dataset.replace('-', '_')}_AllPlusRejected_Confusion_Matrix"
        outputs[f"FigS01{letter}"] = standalone_confusion_figure(
            CONF / CONFUSION_MAP[dataset][1],
            dataset,
            f"{dataset}: all samples with Rejected",
            SUPP_FIG,
            stem,
        )
    return outputs


def fig11_risk_coverage() -> dict[str, str]:
    df = pd.read_csv(MAIN / "risk_coverage_discrete_operating_points.csv")
    df = df[df["dataset"].isin(MAIN4_ORDER)]
    methods = ["kmeans", "any2", "kmeans_distance_reject", "all3"]
    markers = {"kmeans": "o", "any2": "^", "kmeans_distance_reject": "s", "all3": "D"}
    colors = {"kmeans": PALETTE["blue"], "any2": "#4F83B7", "kmeans_distance_reject": PALETTE["red"], "all3": PALETTE["teal"]}
    method_labels = {
        "kmeans": "KMeans",
        "any2": "C2",
        "kmeans_distance_reject": "Distance rejection",
        "all3": "C3",
    }
    dataset_colors = {
        "F_new": PALETTE["gray"],
        "V_new": PALETTE["orange"],
        "M_new": PALETTE["blue"],
        "G_new": PALETTE["teal"],
    }
    fig, ax = plt.subplots(figsize=(9.7, 5.9))
    for ds in MAIN4_ORDER:
        sub = df[df["dataset"] == ds].set_index("method")
        xs = []
        ys = []
        for m in methods:
            if m not in sub.index:
                continue
            r = sub.loc[m]
            x = r["coverage"] * 100
            y = r["selective_risk"] * 100
            xs.append(x)
            ys.append(y)
            ax.scatter(
                x,
                y,
                s=50,
                marker=markers[m],
                color=dataset_colors[ds],
                edgecolor=colors[m],
                linewidth=1.0,
                alpha=0.95,
            )
        if len(xs) > 1:
            ax.plot(xs, ys, color=dataset_colors[ds], linewidth=1.0, alpha=0.55)
    ax.set_xlim(70, 102)
    ax.set_ylim(-0.05, 13.0)
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Selective risk, 1 - Acc_kept (%)")
    ax.grid(color="#E5E7EB", linewidth=0.6)
    dataset_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=7, markerfacecolor=dataset_colors[ds], markeredgecolor="white", label=DATASET_LABELS[ds])
        for ds in MAIN4_ORDER
    ]
    method_handles = [
        Line2D([0], [0], marker=markers[m], linestyle="None", markersize=7, markerfacecolor="white", markeredgecolor=colors[m], markeredgewidth=1.1, label=method_labels[m])
        for m in methods
    ]
    leg1 = ax.legend(handles=dataset_handles, title="Dataset", loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, title_fontsize=7)
    ax.add_artist(leg1)
    ax.legend(handles=method_handles, title="Method", loc="lower left", bbox_to_anchor=(1.01, 0.0), fontsize=7, title_fontsize=7)
    ax.set_title("Discrete risk-coverage operating points")
    fig.subplots_adjust(left=0.09, right=0.76, top=0.90, bottom=0.12)
    return save_figure(fig, MAIN_FIG, "Fig11_Discrete_Risk_Coverage_Points")


def write_inventory(outputs: dict[str, dict[str, str]]) -> None:
    rows = [
        ("Fig01", "Detailed framework", "method schematic", "script-defined method flow", outputs["Fig01"]),
        ("Fig02", "Representative dataset images", "real image plate", "source_data/representative_images", outputs["Fig02"]),
        ("Fig03", "Post-hoc many-to-one mapping", "method schematic", "script-defined schematic", outputs["Fig03"]),
        ("Fig04", "Main C3 results", "bar chart", "source_data/main_results/clean_main.csv", outputs["Fig04"]),
        ("Fig05", "All vs retained clustering metrics", "bar chart", "source_data/main_results/summary_all_vs_kept_cluster_metrics.csv", outputs["Fig05"]),
        ("Fig06", "Fair dimensionality reduction", "bar chart", "source_data/main_results/fair_reduction_5seeds.csv", outputs["Fig06"]),
        ("Fig07", "Parameter sensitivity", "bar chart", "source_data/main_results/parameter_sensitivity_5seeds.csv", outputs["Fig07"]),
        ("Fig08", "External DINOv2 baseline", "errorbar scatter", "source_data/dinov2/summary_convnext_vs_dinov2_key.csv", outputs["Fig08"]),
        ("Fig09", "Weak and high-rejection classes", "horizontal bars", "source_data/confusion_matrices/weak_or_high_rejection_classes_5seed.csv", outputs["Fig09"]),
    ]
    for letter, dataset in zip("abcd", CONFUSION_MAP):
        fig_id = f"Fig10{letter}"
        rows.append(
            (
                fig_id,
                f"{dataset} retained confusion matrix",
                "standalone heatmap",
                f"source_data/confusion_matrices/{CONFUSION_MAP[dataset][0]}",
                outputs[fig_id],
            )
        )
    rows.append(
        ("Fig11", "Discrete risk-coverage points", "scatter/line", "source_data/main_results/risk_coverage_discrete_operating_points.csv", outputs["Fig11"])
    )
    # Keep the supplementary matrices in the same machine-readable inventory.
    for letter, dataset in zip("abcd", CONFUSION_MAP):
        fig_id = f"FigS01{letter}"
        rows.append(
            (
                fig_id,
                f"{dataset} all-sample matrix with Rejected",
                "standalone heatmap",
                f"source_data/confusion_matrices/{CONFUSION_MAP[dataset][1]}",
                outputs[fig_id],
            )
        )
    with (ROOT / "figure_experiment_inventory.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["figure", "title", "type", "source", "png", "pdf", "tif"])
        writer.writeheader()
        for fig_id, title, typ, source, paths in rows:
            writer.writerow({"figure": fig_id, "title": title, "type": typ, "source": source, **paths})


def write_html(outputs: dict[str, dict[str, str]]) -> None:
    cards = []
    main_ids = [f"Fig{i:02d}" for i in range(1, 10)] + ["Fig10a", "Fig10b", "Fig10c", "Fig10d", "Fig11"]
    title_map = {
        "Fig01": "Detailed framework",
        "Fig02": "Representative dataset images",
        "Fig03": "Post-hoc many-to-one mapping",
        "Fig04": "Main C3 results",
        "Fig05": "All vs retained metrics",
        "Fig06": "Fair dimensionality reduction",
        "Fig07": "Parameter sensitivity",
        "Fig08": "External DINOv2 baseline",
        "Fig09": "Weak/high-rejection classes",
        "Fig10a": "PV-Fruit retained confusion matrix",
        "Fig10b": "PV-Vegetable retained confusion matrix",
        "Fig10c": "MCLD-11 retained confusion matrix",
        "Fig10d": "DFLD-BR-4 retained confusion matrix",
        "Fig11": "Discrete risk-coverage points",
    }
    for fig_id in main_ids:
        paths = outputs[fig_id]
        cards.append(
            f"""
            <section class="card">
              <h3>{fig_id}. {html.escape(title_map[fig_id])}</h3>
              <a href="{paths['png']}" target="_blank"><img src="{paths['png']}" alt="{fig_id}"></a>
              <p><a href="{paths['png']}">PNG</a> <a href="{paths['pdf']}">PDF</a> <a href="{paths['tif']}">TIFF</a></p>
            </section>
            """
        )
    supp_cards = []
    supp_titles = {
        "FigS01a": "PV-Fruit all samples with Rejected",
        "FigS01b": "PV-Vegetable all samples with Rejected",
        "FigS01c": "MCLD-11 all samples with Rejected",
        "FigS01d": "DFLD-BR-4 all samples with Rejected",
    }
    for fig_id in ["FigS01a", "FigS01b", "FigS01c", "FigS01d"]:
        paths = outputs[fig_id]
        supp_cards.append(
            f"""
            <section class="card">
              <h3>{fig_id}. {html.escape(supp_titles[fig_id])}</h3>
              <a href="{paths['png']}" target="_blank"><img src="{paths['png']}" alt="{fig_id}"></a>
              <p><a href="{paths['png']}">PNG</a> <a href="{paths['pdf']}">PDF</a> <a href="{paths['tif']}">TIFF</a></p>
            </section>
            """
        )
    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>V13 figure reproduction report</title>
<style>
body{{margin:0;background:#f5f7fa;color:#1f2937;font-family:Arial,'Microsoft YaHei',sans-serif;line-height:1.6}}
header{{background:#fff;border-bottom:1px solid #d9e1ea;padding:24px 32px}}
main{{max-width:1450px;margin:0 auto;padding:20px 24px 44px}}
.note,.card{{background:#fff;border:1px solid #d9e1ea;border-radius:8px;padding:16px;margin:14px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:16px}}
img{{display:block;width:100%;height:auto;border:1px solid #e5e7eb;background:white;border-radius:6px}}
a{{color:#16697a;text-decoration:none;margin-right:10px}} code{{background:#eef3f5;padding:2px 5px;border-radius:4px}}
</style>
</head>
<body>
<header>
<h1>V13 figure reproduction report</h1>
<p>All figures are rebuilt by <code>code/rebuild_v13_figures.py</code> from archived CSV files and representative source images; no values are inferred from manuscript text.</p>
</header>
<main>
<section class="note">
<p>Rerun command: <code>python code/rebuild_v13_figures.py</code></p>
<p>Figure inventory: <a href="figure_experiment_inventory.csv">figure_experiment_inventory.csv</a></p>
</section>
<h2>Main Figures</h2>
<div class="grid">
{''.join(cards)}
</div>
<h2>Supplementary Confusion Matrices</h2>
<div class="grid">
{''.join(supp_cards)}
</div>
</main>
</body>
</html>"""
    (ROOT / "index.html").write_text(html_doc, encoding="utf-8")


def main() -> None:
    prepare_source_data_from_server()
    outputs: dict[str, dict[str, str]] = {}
    outputs["Fig01"] = fig01_framework()
    outputs["Fig02"] = fig02_representative_images()
    outputs["Fig03"] = fig03_mapping_schematic()
    outputs["Fig04"] = fig04_main_results()
    outputs["Fig05"] = fig05_all_vs_retained_metrics()
    outputs["Fig06"] = fig06_fair_reduction()
    outputs["Fig07"] = fig07_parameter_sensitivity()
    outputs["Fig08"] = fig08_dinov2_baseline()
    outputs["Fig09"] = fig09_weak_classes()
    outputs.update(fig10_kept_confusions())
    outputs["Fig11"] = fig11_risk_coverage()
    outputs.update(supplementary_confusions())
    write_inventory(outputs)
    write_html(outputs)
    print(json.dumps({"out": str(OUT), "html": str(ROOT / "index.html")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



