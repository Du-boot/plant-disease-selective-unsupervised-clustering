#!/usr/bin/env python
"""One-command reproduction of the original deploy clustering workflow.

The original scripts are copied into an isolated run directory. Only the
dataset path is adjusted in that copy; ConvNeXt, 100-D UMAP, 20 clusters, and
KMeans/Birch/Agg intersection voting remain the deploy implementation.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


DEPLOY_SOURCE = Path("/data1/D/deploy")
DEFAULT_DATASET = Path("/data1/D/2_new")
DEFAULT_OUTPUT = Path("/data1/D/hostal/deploy_original_2new_100d")
CLASS_NUM = 20
UMAP_DIM = 100

FILES_TO_COPY = [
    "make_umap.py",
    "1_umap_cluster.py",
    "main.py",
    "model.py",
    "model_c.py",
    "config.py",
    "config_c.py",
    "get_final_result.py",
    "check_result.py",
    "get_acc.py",
    "get_acc_km_bh_agg.py",
    "UMAP.py",
]


def replace_once(path: Path, pattern: str, replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.M)
    if count != 1:
        raise RuntimeError(f"Could not update expected setting in {path}")
    path.write_text(updated, encoding="utf-8")


def run(command: list[str], cwd: Path) -> None:
    print("[run]", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def prepare_worktree(dataset: Path, work_dir: Path) -> None:
    if not DEPLOY_SOURCE.is_dir():
        raise FileNotFoundError(f"Original deploy source not found: {DEPLOY_SOURCE}")
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset not found: {dataset}")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    (work_dir / "20250929_129" / "result_umap128").mkdir(parents=True)

    # deploy/model.py and get_final_result.py resolve images as ../<dataset>.
    # Preserve that original convention without duplicating the dataset.
    visible_dataset = work_dir.parent / dataset.name
    if visible_dataset.exists() or visible_dataset.is_symlink():
        if visible_dataset.resolve() != dataset:
            raise RuntimeError(f"{visible_dataset} already exists and is not {dataset}")
    else:
        visible_dataset.symlink_to(dataset, target_is_directory=True)

    for name in FILES_TO_COPY:
        shutil.copy2(DEPLOY_SOURCE / name, work_dir / name)
    shutil.copy2(DEPLOY_SOURCE / "cn" / "encode.py", work_dir / "encode.py")

    # These are the only source settings adapted for the isolated dataset run.
    replace_once(work_dir / "encode.py", r'^\s*train_path\s*=.*$', f'    train_path = "{dataset}/"')
    replace_once(work_dir / "encode.py", r'^\s*out_dir\s*=.*$', '    out_dir = "encode_2048"')
    replace_once(work_dir / "get_final_result.py", r'^\s*folder\s*=.*$', f'    folder = "{dataset.name}"')
    replace_once(work_dir / "config_c.py", r'^\s*num\s*=.*$', f'num = {CLASS_NUM}    # original deploy cluster count')
    replace_once(work_dir / "make_umap.py", r'^\s*k_lst=\[.*?\].*$', f'    k_lst=[{UMAP_DIM}]  # UMAP dimension for this isolated run')
    # The original 1_umap_cluster.py calls classes_result() without forwarding
    # class_num, which silently falls back to 20 and breaks K != 20 runs.
    replace_once(
        work_dir / "1_umap_cluster.py",
        r's_acc\s*=\s*classes_result\(\)',
        's_acc = classes_result(class_num)',
    )


def make_confusion_matrix(work_dir: Path) -> dict:
    root = work_dir / "20250929_129" / f"merged_result_final{CLASS_NUM}"
    if not root.is_dir():
        raise FileNotFoundError(f"Missing original merged output: {root}")
    y_true, y_pred, rest = [], [], 0
    for predicted_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if predicted_dir.name == "rest":
            rest += sum(path.is_file() for path in predicted_dir.iterdir())
            continue
        try:
            predicted = int(predicted_dir.name)
        except ValueError:
            continue
        for image in predicted_dir.iterdir():
            if not image.is_file():
                continue
            try:
                y_true.append(int(image.name.split("_", 1)[0]))
                y_pred.append(predicted)
            except ValueError:
                continue
    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    correct = int(np.trace(matrix))
    total = len(y_true) + rest
    metrics = {
        "total": total,
        "used": len(y_true),
        "rest": rest,
        "correct": correct,
        "acc_kept": correct / len(y_true) if y_true else 0.0,
        "drop": rest / total if total else 0.0,
        "acc_all_if_rest_wrong": correct / total if total else 0.0,
        "labels": labels,
        "confusion_matrix": matrix.tolist(),
    }
    np.savetxt(work_dir / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Deploy original: {UMAP_DIM}D UMAP, {CLASS_NUM} clusters")
    fig.tight_layout()
    fig.savefig(work_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)
    return metrics


def main() -> None:
    global CLASS_NUM, UMAP_DIM
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--class-num", type=int, default=CLASS_NUM)
    parser.add_argument("--umap-dim", type=int, default=UMAP_DIM)
    parser.add_argument(
        "--reuse-features-from",
        type=Path,
        help="Reuse encode_2048/ and result_umap_17255_100_2/ from a completed compatible run.",
    )
    parser.add_argument(
        "--reuse-encode-from",
        type=Path,
        help="Reuse only encode_2048/ from a completed compatible dataset run, then recompute UMAP.",
    )
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    output = args.output.resolve()
    CLASS_NUM = args.class_num
    UMAP_DIM = args.umap_dim
    prepare_worktree(dataset, output)
    if args.reuse_features_from and args.reuse_encode_from:
        raise ValueError("Use only one of --reuse-features-from or --reuse-encode-from")
    if args.reuse_features_from:
        feature_source = args.reuse_features_from.resolve()
        for name in ("encode_2048", "result_umap_17255_100_2"):
            source = feature_source / name
            if not source.is_dir():
                raise FileNotFoundError(f"Reusable feature directory not found: {source}")
            shutil.copytree(source, output / name)
    else:
        if args.reuse_encode_from:
            encode_source = args.reuse_encode_from.resolve() / "encode_2048"
            if not encode_source.is_dir():
                raise FileNotFoundError(f"Reusable encode directory not found: {encode_source}")
            shutil.copytree(encode_source, output / "encode_2048")
        else:
            run([sys.executable, "encode.py"], output)
        run([
            sys.executable,
            "-c",
            "from make_umap import main as umap_main; umap_main('encode_2048/1_2048.txt', 0)",
        ], output)
    reduced = output / "result_umap_17255_100_2" / "1_2048.txt"
    target = output / "20250929_129" / "result_umap128" / "1_128.txt"
    shutil.copy2(reduced, target)
    run([sys.executable, "1_umap_cluster.py"], output)
    metrics = make_confusion_matrix(output)
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[metrics]", metrics, flush=True)


if __name__ == "__main__":
    main()
