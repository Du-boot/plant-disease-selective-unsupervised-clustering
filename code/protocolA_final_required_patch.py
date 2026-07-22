#!/usr/bin/env python3
"""Required final-analysis patch for Protocol A experiments.

This script does not change the clustering protocol. It only adds the pieces
needed for manuscript reporting:

- class-level rejection/acc averaged across the five main seeds;
- seed-level paired bootstrap confidence intervals;
- average row-normalized confusion matrices over five seeds;
- a discrete risk-coverage table with explicit wording;
- a pHash connected-component cleanup template for manual audit results;
- a short M-cleaning note grounded in the available label_map.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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


def m_cleaning_note(data_root: Path, out: Path) -> None:
    out_dir = out / "m_cleaning_note"
    out_dir.mkdir(parents=True, exist_ok=True)
    label_map = data_root / "M_new_drop5_drop7" / "label_map.csv"
    retained = []
    if label_map.exists():
        with label_map.open(encoding="utf-8-sig", newline="") as fh:
            retained = list(csv.DictReader(fh))
    retained_old = [r.get("old_label", "") for r in retained]
    note = f"""# M_new_drop5_drop7 quality-control note

This note is intentionally conservative. The cleaned M dataset should not be
justified by improved clustering results alone.

Available label_map: `{label_map}`

Retained old_label values found in label_map:

{", ".join(retained_old) if retained_old else "label_map not found"}

Recommended manuscript wording:

> The quality-controlled M subset is reported as the main stress-test set,
> while the complete M_new set is retained as a robustness comparison. The
> exclusion criteria must be based on dataset-level quality-control criteria
> such as ambiguous class definition, non-leaf disease images, label noise,
> corrupted files, insufficient samples, or semantic duplication, rather than
> model performance.

Important limitation:

> Compared with complete M_new, the cleaned subset improves post-hoc aligned
> retained-sample accuracy and coverage, but some structure metrics such as
> ARI/NMI do not improve monotonically. Therefore the cleaned set should be
> described as a quality-controlled subset, not as evidence that clustering
> quality improved in every metric.
"""
    (out_dir / "m_cleaning_note.md").write_text(note, encoding="utf-8")
    if retained:
        write_csv(out_dir / "M_new_drop5_drop7_retained_label_map.csv", retained)


def phash_component_template(out: Path) -> None:
    script = r'''#!/usr/bin/env python3
"""Apply filled pHash manual audit decisions by connected components.

Default mode is non-destructive. It writes a removal manifest and, optionally,
copies cleaned datasets to a new directory. Fill the audit CSV columns first:

- judgement: exact_variant, crop_or_rotation, same_leaf_sequence,
  visually_similar, uncertain
- action: delete_one, delete_a, delete_b, keep_all

For delete_one, the script builds connected components and keeps one image per
component by largest file size, then lexical name. It never deletes originals.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict, deque
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def choose_keep(data_root: Path, ds: str, names: set[str]) -> str:
    def rank(name: str):
        p = data_root / ds / name
        size = p.stat().st_size if p.exists() else -1
        return (-size, name)
    return sorted(names, key=rank)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-csv", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("/data1/D"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--copy-clean-datasets", action="store_true")
    args = ap.parse_args()

    rows = read_rows(args.audit_csv)
    graph = defaultdict(lambda: defaultdict(set))
    forced_remove = defaultdict(set)
    kept_pairs = 0
    for r in rows:
        ds = r.get("dataset", "")
        a = Path(r.get("image_a", "")).name
        b = Path(r.get("image_b", "")).name
        action = r.get("action", "").strip().lower()
        if action in {"keep_all", "keep", ""}:
            kept_pairs += 1
            continue
        if action == "delete_a":
            forced_remove[ds].add(a)
        elif action == "delete_b":
            forced_remove[ds].add(b)
        elif action == "delete_one":
            graph[ds][a].add(b)
            graph[ds][b].add(a)
        else:
            raise ValueError(f"Unknown action {action!r} in row {r}")

    manifest = []
    for ds, adj in graph.items():
        seen = set()
        for start in sorted(adj):
            if start in seen:
                continue
            q = deque([start])
            comp = set()
            seen.add(start)
            while q:
                cur = q.popleft()
                comp.add(cur)
                for nxt in adj[cur]:
                    if nxt not in seen:
                        seen.add(nxt)
                        q.append(nxt)
            keep = choose_keep(args.data_root, ds, comp)
            for name in sorted(comp - {keep}):
                forced_remove[ds].add(name)
                manifest.append({"dataset": ds, "component_keep": keep, "remove_file": name, "reason": "phash_component_delete_one"})

    for ds, names in forced_remove.items():
        for name in sorted(names):
            if not any(m["dataset"] == ds and m["remove_file"] == name for m in manifest):
                manifest.append({"dataset": ds, "component_keep": "", "remove_file": name, "reason": "manual_forced_remove"})

    write_rows(args.out / "phash_removal_manifest.csv", manifest)

    if args.copy_clean_datasets:
        for ds_dir in sorted(p for p in args.data_root.iterdir() if p.is_dir()):
            ds = ds_dir.name
            remove = forced_remove.get(ds, set())
            if not remove:
                continue
            target = args.out / f"{ds}_phash_clean"
            target.mkdir(parents=True, exist_ok=True)
            for p in sorted(ds_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and p.name not in remove:
                    shutil.copy2(p, target / p.name)

    print(f"manual keep/blank pairs: {kept_pairs}")
    print(f"removal candidates: {len(manifest)}")
    print(args.out)


if __name__ == "__main__":
    main()
'''
    tool_dir = out / "tools"
    tool_dir.mkdir(parents=True, exist_ok=True)
    (tool_dir / "apply_phash_connected_components.py").write_text(script, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("/data1/D"))
    ap.add_argument("--feature-root", type=Path, default=Path("/data1/D/hostal/audit_unsupervised_20260720"))
    ap.add_argument("--followup-root", type=Path, default=Path("/data1/D/hostal/protocolA_followup_20260720"))
    ap.add_argument("--out", type=Path, default=Path("/data1/D/hostal/protocolA_final_required_patch_20260720"))
    args = ap.parse_args()

    class_level_5seed(args.feature_root, args.followup_root, args.out)
    paired_seed_bootstrap(args.followup_root, args.out)
    discrete_risk_coverage(args.followup_root, args.out)
    m_cleaning_note(args.data_root, args.out)
    phash_component_template(args.out)
    (args.out / "README.md").write_text(textwrap.dedent(f"""
        # Protocol A final required patch

        Generated additions:

        - `class_level_5seed/`: class-level rejection and acc across five seeds,
          plus five-seed averaged row-normalized confusion matrices.
        - `bootstrap_seedlevel/`: paired seed-level bootstrap confidence intervals.
        - `risk_coverage_discrete/`: discrete risk-coverage operating points.
        - `m_cleaning_note/`: conservative M-cleaning wording and retained label map.
        - `tools/apply_phash_connected_components.py`: non-destructive utility to
          apply filled pHash manual decisions after human review.

        pHash manual audit is still not closed until the `judgement` and `action`
        fields are filled by human reviewers.
    """).strip() + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
