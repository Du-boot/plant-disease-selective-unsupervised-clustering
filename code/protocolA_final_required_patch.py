#!/usr/bin/env python3
"""Required final-analysis patch for Protocol A experiments.

This script does not change the clustering protocol. It only adds the pieces
needed for manuscript reporting:

- class-level rejection/acc averaged across the five main seeds;
- seed-level paired bootstrap confidence intervals;
- average row-normalized confusion matrices over five seeds;
- a discrete risk-coverage table with explicit wording;

The current final manuscript protocol uses complete M_new (MCLD-11). pHash is
treated only as a near-duplicate risk audit and is not used for automatic
deletion or rerunning experiments.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import textwrap
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

from protocolA_final_supplements import (
    DATASETS,
    DISPLAY_NAMES,
    SEEDS,
    fit_or_load_main,
    load_clean_view,
    map_predict,
    read_csv,
    save_heatmap_html,
    write_csv,
    write_matrix_csv,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(os.environ.get("PROTOCOLA_DATA_ROOT", str(REPO_ROOT / "data" / "images_clean")))
DEFAULT_FEATURE_ROOT = Path(os.environ.get("PROTOCOLA_FEATURE_ROOT", str(REPO_ROOT / "outputs" / "convnext_features")))
DEFAULT_FOLLOWUP_ROOT = Path(os.environ.get("PROTOCOLA_FOLLOWUP_ROOT", str(REPO_ROOT / "outputs" / "protocolA_followup")))
DEFAULT_OUT_ROOT = Path(os.environ.get("PROTOCOLA_REQUIRED_ROOT", str(REPO_ROOT / "outputs" / "protocolA_final_required_patch")))


def pct(x: float) -> str:
    return "NA" if not math.isfinite(x) else f"{x * 100:.2f}%"


def f(v: str | float | int | None) -> float:
    if v is None or v == "" or str(v).lower() in {"nan", "none"}:
        return math.nan
    return float(v)


def mean(vals: list[float]) -> float:
    vals = [x for x in vals if math.isfinite(x)]
    return float(np.mean(vals)) if vals else math.nan


def std(vals: list[float]) -> float:
    vals = [x for x in vals if math.isfinite(x)]
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0 if vals else math.nan


def class_level_5seed(feature_root: Path, followup_root: Path, out: Path) -> None:
    audit_root = followup_root / "duplicate_audit"
    out_dir = out / "class_level_5seed"
    out_dir.mkdir(parents=True, exist_ok=True)
    long_rows: list[dict] = []

    for ds in DATASETS:
        _, y, _ = load_clean_view(feature_root, audit_root, ds)
        labels = sorted(set(y.tolist()))
        display = DISPLAY_NAMES[ds]
        all_norm_mats: list[np.ndarray] = []
        kept_norm_mats: list[np.ndarray] = []

        for seed in SEEDS:
            _, _, aligned, votes = fit_or_load_main(followup_root, None, ds, seed)
            keep = np.all(votes == votes[0:1], axis=0)
            clusters = aligned["kmeans"]
            pred, _ = map_predict(y, clusters, keep)

            kept_counts = np.zeros((len(labels), len(labels)), dtype=float)
            all_counts = np.zeros((len(labels), len(labels) + 1), dtype=float)
            label_to_row = {c: i for i, c in enumerate(labels)}
            label_to_col = {c: i for i, c in enumerate(labels)}

            for c in labels:
                class_mask = y == c
                kept_mask = class_mask & keep
                total = int(class_mask.sum())
                kept = int(kept_mask.sum())
                correct = int(np.sum(pred[kept_mask] == y[kept_mask])) if kept else 0
                long_rows.append({
                    "dataset": ds,
                    "seed": seed,
                    "class_id": c,
                    "class_name": display.get(c, str(c)),
                    "total": total,
                    "kept": kept,
                    "rejected": total - kept,
                    "rejection_rate": (total - kept) / total if total else math.nan,
                    "acc_kept": correct / kept if kept else math.nan,
                })

            for i in range(len(y)):
                r = label_to_row[int(y[i])]
                if not keep[i] or pred[i] < 0:
                    all_counts[r, -1] += 1
                else:
                    c = label_to_col.get(int(pred[i]))
                    if c is None:
                        all_counts[r, -1] += 1
                    else:
                        all_counts[r, c] += 1
                        kept_counts[r, c] += 1

            all_norm = all_counts / np.maximum(all_counts.sum(axis=1, keepdims=True), 1)
            kept_norm = kept_counts / np.maximum(kept_counts.sum(axis=1, keepdims=True), 1)
            all_norm_mats.append(all_norm)
            kept_norm_mats.append(kept_norm)

        row_names = [display.get(c, str(c)) for c in labels]
        col_names_all = row_names + ["Rejected"]
        avg_all = np.mean(all_norm_mats, axis=0)
        avg_kept = np.mean(kept_norm_mats, axis=0)
        write_matrix_csv(out_dir / f"{ds}_5seed_avg_all_confusion_with_rejected_row_normalized.csv", avg_all, col_names_all, row_names)
        write_matrix_csv(out_dir / f"{ds}_5seed_avg_kept_confusion_row_normalized.csv", avg_kept, row_names, row_names)
        save_heatmap_html(out_dir / f"{ds}_5seed_avg_all_confusion_with_rejected.html", avg_all, col_names_all)
        save_heatmap_html(out_dir / f"{ds}_5seed_avg_kept_confusion.html", avg_kept, row_names)

    write_csv(out_dir / "class_level_rejection_5seed_long.csv", long_rows)
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in long_rows:
        grouped[(r["dataset"], int(r["class_id"]))].append(r)
    summary_rows = []
    for (ds, cid), rows in sorted(grouped.items()):
        summary_rows.append({
            "dataset": ds,
            "class_id": cid,
            "class_name": rows[0]["class_name"],
            "total": rows[0]["total"],
            "kept_mean": mean([f(r["kept"]) for r in rows]),
            "kept_std": std([f(r["kept"]) for r in rows]),
            "rejected_mean": mean([f(r["rejected"]) for r in rows]),
            "rejected_std": std([f(r["rejected"]) for r in rows]),
            "rejection_rate_mean": mean([f(r["rejection_rate"]) for r in rows]),
            "rejection_rate_std": std([f(r["rejection_rate"]) for r in rows]),
            "acc_kept_mean": mean([f(r["acc_kept"]) for r in rows]),
            "acc_kept_std": std([f(r["acc_kept"]) for r in rows]),
            "n_seeds": len(rows),
        })
    write_csv(out_dir / "class_level_rejection_5seed_summary.csv", summary_rows)

    weak = sorted(
        [r for r in summary_rows if r["dataset"] != "G_new"],
        key=lambda r: (f(r["acc_kept_mean"]), -f(r["rejection_rate_mean"])),
    )[:20]
    write_csv(out_dir / "weak_or_high_rejection_classes_5seed.csv", weak)


def paired_seed_bootstrap(followup_root: Path, out: Path, reps: int = 20000) -> None:
    main = read_csv(followup_root / "followup_results" / "clean_main.csv")
    rnd = read_csv(followup_root / "followup_results" / "random_same_coverage_completed.csv")
    rng = np.random.RandomState(20260720)
    metrics = ["acc_kept", "ari_kept", "nmi_kept"]
    comparisons = [
        ("all3", "kmeans"),
        ("all3", "any2"),
        ("all3", "kmeans_distance_reject"),
        ("all3", "random_same_coverage"),
    ]
    rows = []
    for ds in DATASETS:
        for a, b in comparisons:
            for metric in metrics:
                left = {
                    int(r["seed"]): f(r[metric])
                    for r in main
                    if r["dataset"] == ds and r["method"] == a and math.isfinite(f(r.get(metric)))
                }
                if b == "random_same_coverage":
                    tmp: dict[int, list[float]] = defaultdict(list)
                    for r in rnd:
                        if r["dataset"] == ds and math.isfinite(f(r.get(metric))):
                            tmp[int(r["seed"])].append(f(r[metric]))
                    right = {s: mean(v) for s, v in tmp.items()}
                    right_note = "mean_over_random_rejection_repeats_within_seed"
                else:
                    right = {
                        int(r["seed"]): f(r[metric])
                        for r in main
                        if r["dataset"] == ds and r["method"] == b and math.isfinite(f(r.get(metric)))
                    }
                    right_note = "same_seed_method_pair"
                seeds = sorted(set(left) & set(right))
                diffs = np.asarray([left[s] - right[s] for s in seeds], dtype=float)
                if len(diffs) == 0:
                    continue
                boots = np.asarray([np.mean(diffs[rng.randint(0, len(diffs), len(diffs))]) for _ in range(reps)])
                lo, hi = np.percentile(boots, [2.5, 97.5])
                rows.append({
                    "dataset": ds,
                    "comparison": f"{a} - {b}",
                    "metric": metric,
                    "sampling_unit": "random_seed",
                    "n_seed_pairs": len(diffs),
                    "random_rejection_handling": right_note,
                    "bootstrap_reps": reps,
                    "ci_method": "percentile",
                    "mean_diff": float(np.mean(diffs)),
                    "seed_diff_std": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
                    "ci95_low": float(lo),
                    "ci95_high": float(hi),
                    "ci_excludes_zero": int(lo > 0 or hi < 0),
                    "practical_threshold": 0.005,
                    "practically_meaningful": int(abs(float(np.mean(diffs))) >= 0.005),
                    "seed_diffs": ";".join(f"{x:.8f}" for x in diffs),
                })
    write_csv(out / "bootstrap_seedlevel" / "method_difference_seedlevel_bootstrap_ci.csv", rows)


def discrete_risk_coverage(followup_root: Path, out: Path) -> None:
    main = read_csv(followup_root / "followup_results" / "clean_main.csv")
    rows = []
    for ds in DATASETS:
        for method in ["kmeans", "any2", "all3", "kmeans_distance_reject"]:
            sub = [r for r in main if r["dataset"] == ds and r["method"] == method]
            if not sub:
                continue
            acc = mean([f(r["acc_kept"]) for r in sub])
            rej = mean([f(r["rejection_rate"]) for r in sub])
            rows.append({
                "dataset": ds,
                "method": method,
                "analysis_type": "discrete_operating_point_not_continuous_AURC",
                "coverage": 1 - rej,
                "selective_risk": 1 - acc,
                "acc_kept": acc,
                "rejection_rate": rej,
                "n_seeds": len(sub),
            })
    write_csv(out / "risk_coverage_discrete" / "risk_coverage_discrete_operating_points.csv", rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    ap.add_argument("--followup-root", type=Path, default=DEFAULT_FOLLOWUP_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_ROOT)
    args = ap.parse_args()
    print("[config]")
    print(f"  data_root     = {args.data_root}")
    print(f"  feature_root  = {args.feature_root}")
    print(f"  followup_root = {args.followup_root}")
    print(f"  out           = {args.out}")

    class_level_5seed(args.feature_root, args.followup_root, args.out)
    paired_seed_bootstrap(args.followup_root, args.out)
    discrete_risk_coverage(args.followup_root, args.out)
    (args.out / "README.md").write_text(textwrap.dedent(f"""
        # Protocol A final required patch

        Generated additions:

        - `class_level_5seed/`: class-level rejection and acc across five seeds,
          plus five-seed averaged row-normalized confusion matrices.
        - `bootstrap_seedlevel/`: paired seed-level bootstrap confidence intervals.
        - `risk_coverage_discrete/`: discrete risk-coverage operating points.

        Final dataset policy:

        - Main MCLD dataset is complete `M_new` / `MCLD-11`.
        - pHash is only a near-duplicate risk audit and does not delete images.
    """).strip() + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()




