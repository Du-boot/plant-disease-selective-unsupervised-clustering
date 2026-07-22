#!/usr/bin/env python3
"""Final supplements for Protocol A report.

Outputs:
- pHash<=4 human-audit contact sheets and blank judgement CSV.
- class-level rejection tables and confusion matrices for exact-dedup main runs.
- seed/bootstrap confidence intervals for method differences.
- risk-coverage points.
- optional M_new full-dataset main configuration experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, Birch, KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    confusion_matrix,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)


DATASETS = ["F_new", "V_new", "M_new_drop5_drop7", "G_new"]
SEEDS = [11, 22, 33, 44, 55]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DISPLAY_NAMES = {
    "F_new": {
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
    "V_new": {
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
    "M_new_drop5_drop7": {
        0: "Cashew - leaf miner",
        1: "Cashew - red rust",
        2: "Corn - leaf blight",
        3: "Corn - streak virus",
        4: "Potato - fungi",
        5: "Rice - bacterial leaf blight",
        6: "Rice - brown spot",
        7: "Tomato - septoria leaf spot",
        8: "Tomato - verticulium wilt",
    },
    "G_new": {
        0: "Gourd",
        1: "Hibiscus",
        2: "Papaya",
        3: "Zucchini",
    },
    "M_new": {
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
        10: "Tomato - verticulium wilt",
    },
}


def log(msg: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg, flush=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k, v in r.items() if not isinstance(v, (dict, list, np.ndarray))})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def image_files(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def labels_from_names(names: list[str]) -> np.ndarray:
    return np.asarray([int(Path(n).name.split("_", 1)[0]) for n in names], dtype=np.int64)


def load_features(feature_root: Path, ds: str) -> tuple[np.ndarray, list[str]]:
    X = np.load(feature_root / "features" / f"{ds}.npy", mmap_mode="r")
    names = json.loads((feature_root / "features" / f"{ds}.names.json").read_text(encoding="utf-8"))
    return X, names


def load_clean_view(feature_root: Path, audit_root: Path, ds: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X, names = load_features(feature_root, ds)
    keep_map = json.loads((audit_root / "exact_clean_keep_names.json").read_text(encoding="utf-8"))
    keep = set(keep_map[ds])
    idx = [i for i, n in enumerate(names) if n in keep]
    names = [names[i] for i in idx]
    return X[idx], labels_from_names(names), names


def align_to_reference(ref: np.ndarray, other: np.ndarray, k: int) -> np.ndarray:
    overlap = np.zeros((k, k), dtype=np.int64)
    for a, b in zip(ref, other):
        if 0 <= a < k and 0 <= b < k:
            overlap[int(a), int(b)] += 1
    rows, cols = linear_sum_assignment(-overlap)
    mapping = {int(c): int(r) for r, c in zip(rows, cols)}
    return np.asarray([mapping.get(int(x), -1) for x in other], dtype=np.int64)


def map_predict(y: np.ndarray, clusters: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    mapping = {}
    for c in sorted(set(clusters[keep].tolist())):
        mask = keep & (clusters == c)
        if mask.any():
            mapping[int(c)] = int(Counter(y[mask].tolist()).most_common(1)[0][0])
    pred = np.asarray([mapping.get(int(c), -1) for c in clusters], dtype=np.int64)
    return pred, mapping


def score_external(y: np.ndarray, clusters: np.ndarray, keep: np.ndarray, method: str, meta: dict) -> dict:
    keep = keep & (clusters >= 0)
    pred, _ = map_predict(y, clusters, keep)
    n = len(y)
    used = int(keep.sum())
    correct = int(np.sum(pred[keep] == y[keep])) if used else 0
    rejected = n - used
    rejected_correct = int(np.sum(pred[~keep] == y[~keep])) if rejected else 0
    kept_err = 1 - correct / used if used else math.nan
    rej_err = 1 - rejected_correct / rejected if rejected else math.nan
    return {
        **meta,
        "method": method,
        "n": n,
        "kept": used,
        "rejected": rejected,
        "rejection_rate": rejected / n if n else math.nan,
        "acc_kept": correct / used if used else math.nan,
        "overall_accuracy": correct / n if n else math.nan,
        "kept_error": kept_err,
        "rejected_error": rej_err,
        "error_gap": rej_err - kept_err if math.isfinite(kept_err) and math.isfinite(rej_err) else math.nan,
        "ari_all": adjusted_rand_score(y, clusters),
        "nmi_all": normalized_mutual_info_score(y, clusters),
        "ami_all": adjusted_mutual_info_score(y, clusters),
        "ari_kept": adjusted_rand_score(y[keep], clusters[keep]) if used > 1 else math.nan,
        "nmi_kept": normalized_mutual_info_score(y[keep], clusters[keep]) if used > 1 else math.nan,
        "ami_kept": adjusted_mutual_info_score(y[keep], clusters[keep]) if used > 1 else math.nan,
        "homogeneity_kept": homogeneity_score(y[keep], clusters[keep]) if used > 1 else math.nan,
        "completeness_kept": completeness_score(y[keep], clusters[keep]) if used > 1 else math.nan,
        "v_measure_kept": v_measure_score(y[keep], clusters[keep]) if used > 1 else math.nan,
    }


def fit_or_load_main(out_root: Path, X: np.ndarray, ds: str, seed: int, k: int = 20):
    emb = out_root / "cache" / "embeddings" / f"{ds}__exact_dedup_clean__umap100__seed{seed}.npy"
    cl = out_root / "cache" / "clusters" / f"{ds}__exact_dedup_clean__umap100__k{k}__seed{seed}.npz"
    if not emb.exists() or not cl.exists():
        raise FileNotFoundError(f"missing cached main result for {ds} seed {seed}")
    raw_npz = np.load(cl)
    raw = {name: raw_npz[name] for name in raw_npz.files}
    aligned = {
        "kmeans": raw["kmeans"],
        "birch": align_to_reference(raw["kmeans"], raw["birch"], k),
        "agg": align_to_reference(raw["kmeans"], raw["agg"], k),
    }
    votes = np.vstack([aligned["kmeans"], aligned["birch"], aligned["agg"]])
    return np.load(emb, mmap_mode="r"), raw, aligned, votes


def build_phash_audit(data_root: Path, followup_root: Path, out: Path, max_pairs_per_page: int = 24) -> None:
    rows = read_csv(followup_root / "duplicate_audit" / "near_duplicate_pairs.csv")
    rows = [r for r in rows if r.get("highly_suspicious") == "1"]
    audit_rows = []
    img_out = out / "phash_manual_audit"
    img_out.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    thumb = (150, 150)
    gap = 18
    row_h = 200
    page_w = 760
    for ds in DATASETS:
        ds_rows = [r for r in rows if r["dataset"] == ds]
        for i, r in enumerate(ds_rows):
            audit_rows.append({
                "dataset": ds,
                "pair_id": f"{ds}_{i:04d}",
                "image_a": r["file_a"],
                "image_b": r["file_b"],
                "phash_distance": r["phash_hamming"],
                "judgement": "",
                "action": "",
                "notes": "",
            })
        pages = [ds_rows[i:i + max_pairs_per_page] for i in range(0, len(ds_rows), max_pairs_per_page)]
        for pi, page in enumerate(pages, 1):
            canvas = Image.new("RGB", (page_w, max(1, len(page)) * row_h + 40), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text((10, 10), f"{ds} pHash<=4 manual audit page {pi}/{len(pages)}", fill=(0, 0, 0), font=font)
            for ri, r in enumerate(page):
                y0 = 35 + ri * row_h
                for side, fname, x0 in [("A", r["file_a"], 10), ("B", r["file_b"], 180)]:
                    path = data_root / ds / fname
                    try:
                        im = Image.open(path).convert("RGB")
                        im.thumbnail(thumb)
                    except Exception:
                        im = Image.new("RGB", thumb, (230, 230, 230))
                    canvas.paste(im, (x0, y0))
                    draw.text((x0, y0 + 154), f"{side}: {fname[:32]}", fill=(0, 0, 0), font=font)
                draw.text((360, y0), f"id: {ds}_{(pi - 1) * max_pairs_per_page + ri:04d}", fill=(0, 0, 0), font=font)
                draw.text((360, y0 + 18), f"distance: {r['phash_hamming']}", fill=(0, 0, 0), font=font)
                draw.text((360, y0 + 45), "judgement: same_resize | same_crop_rotate | burst_same_leaf | similar_not_duplicate", fill=(0, 0, 0), font=font)
                draw.text((360, y0 + 70), "action: delete_b | keep_both | manual_check", fill=(0, 0, 0), font=font)
            canvas.save(img_out / f"{ds}_phash_le4_page_{pi:02d}.jpg", quality=92)
    write_csv(img_out / "phash_le4_manual_audit_todo.csv", audit_rows)


def class_level_and_matrices(feature_root: Path, followup_root: Path, out: Path, seed: int = 11) -> None:
    audit_root = followup_root / "duplicate_audit"
    rows_class = []
    out_dir = out / "class_level_matrices"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        _, y, names = load_clean_view(feature_root, audit_root, ds)
        _, _, aligned, votes = fit_or_load_main(followup_root, None, ds, seed)
        keep = np.all(votes == votes[0:1], axis=0)
        clusters = aligned["kmeans"]
        pred, mapping = map_predict(y, clusters, keep)
        labels = sorted(set(y.tolist()))
        display = DISPLAY_NAMES[ds]
        for c in labels:
            m = y == c
            mk = m & keep
            total = int(m.sum())
            kept = int(mk.sum())
            correct = int(np.sum(pred[mk] == y[mk])) if kept else 0
            rows_class.append({
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
        cm_kept = confusion_matrix(y[keep], pred[keep], labels=labels)
        cm_all = confusion_matrix(y, np.where(keep, pred, len(labels)), labels=labels + [len(labels)])
        write_matrix_csv(out_dir / f"{ds}_seed{seed}_kept_confusion_counts.csv", cm_kept, [display.get(c, str(c)) for c in labels], [display.get(c, str(c)) for c in labels])
        write_matrix_csv(out_dir / f"{ds}_seed{seed}_all_confusion_with_rejected_counts.csv", cm_all, [display.get(c, str(c)) for c in labels] + ["Rejected"], [display.get(c, str(c)) for c in labels] + ["Rejected"])
        cm_all_norm = cm_all.astype(float)
        denom = cm_all_norm.sum(axis=1, keepdims=True)
        cm_all_norm = np.divide(cm_all_norm, denom, out=np.zeros_like(cm_all_norm), where=denom != 0)
        write_matrix_csv(out_dir / f"{ds}_seed{seed}_all_confusion_with_rejected_row_normalized.csv", cm_all_norm, [display.get(c, str(c)) for c in labels] + ["Rejected"], [display.get(c, str(c)) for c in labels] + ["Rejected"])
        save_heatmap_html(out_dir / f"{ds}_seed{seed}_all_confusion_with_rejected.html", cm_all_norm, [display.get(c, str(c)) for c in labels] + ["Rejected"])
    write_csv(out_dir / "class_level_rejection_seed11.csv", rows_class)


def write_matrix_csv(path: Path, mat: np.ndarray, cols: list[str], rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true_class"] + cols)
        for name, row in zip(rows, mat):
            w.writerow([name] + list(row))


def save_heatmap_html(path: Path, mat: np.ndarray, labels: list[str]) -> None:
    def cell(v):
        intensity = max(0, min(1, float(v)))
        bg = f"rgba(37,111,104,{0.08 + 0.78 * intensity:.3f})"
        txt = f"{v * 100:.1f}%"
        return f'<td style="background:{bg}">{txt}</td>'
    header = "".join(f"<th>{escape(x)}</th>" for x in labels)
    body = []
    for lab, row in zip(labels, mat):
        body.append("<tr>" + f"<th>{escape(lab)}</th>" + "".join(cell(v) for v in row) + "</tr>")
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><style>
body{{font-family:Arial,'Microsoft YaHei',sans-serif;margin:20px;color:#1f2933}}
table{{border-collapse:collapse;font-size:12px}}th,td{{border:1px solid #d8dee8;padding:6px 8px;text-align:center;white-space:nowrap}}
th{{background:#eef2f6}}td{{min-width:58px}}
</style></head><body><table><thead><tr><th>true\\pred</th>{header}</tr></thead><tbody>{''.join(body)}</tbody></table></body></html>"""
    path.write_text(html, encoding="utf-8")


def escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))


def bootstrap_ci(followup_root: Path, out: Path, reps: int = 10000) -> None:
    main = read_csv(followup_root / "followup_results" / "clean_main.csv")
    rnd = read_csv(followup_root / "followup_results" / "random_same_coverage_completed.csv")
    comparisons = [
        ("all3", "kmeans"),
        ("all3", "any2"),
        ("all3", "kmeans_distance_reject"),
        ("all3", "random_same_coverage"),
    ]
    metrics = ["acc_kept", "ari_kept", "nmi_kept"]
    rng = np.random.RandomState(20260720)
    rows = []
    for ds in DATASETS:
        for a, b in comparisons:
            for metric in metrics:
                if b == "random_same_coverage":
                    all3 = {int(r["seed"]): float(r[metric]) for r in main if r["dataset"] == ds and r["method"] == "all3"}
                    diffs = [all3[int(r["seed"])] - float(r[metric]) for r in rnd if r["dataset"] == ds and r[metric] not in ("", "nan")]
                else:
                    left = {int(r["seed"]): float(r[metric]) for r in main if r["dataset"] == ds and r["method"] == a and r[metric] not in ("", "nan")}
                    right = {int(r["seed"]): float(r[metric]) for r in main if r["dataset"] == ds and r["method"] == b and r[metric] not in ("", "nan")}
                    seeds = sorted(set(left) & set(right))
                    diffs = [left[s] - right[s] for s in seeds]
                if not diffs:
                    continue
                diffs = np.asarray(diffs, dtype=float)
                boots = np.asarray([np.mean(diffs[rng.randint(0, len(diffs), len(diffs))]) for _ in range(reps)])
                lo, hi = np.percentile(boots, [2.5, 97.5])
                rows.append({
                    "dataset": ds,
                    "comparison": f"{a} - {b}",
                    "metric": metric,
                    "n_pairs": len(diffs),
                    "mean_diff": float(np.mean(diffs)),
                    "ci95_low": float(lo),
                    "ci95_high": float(hi),
                    "ci_excludes_zero": int(lo > 0 or hi < 0),
                })
    write_csv(out / "bootstrap_ci" / "method_difference_bootstrap_ci.csv", rows)


def risk_coverage(followup_root: Path, out: Path) -> None:
    main = read_csv(followup_root / "followup_results" / "clean_main.csv")
    methods = ["kmeans", "any2", "all3", "kmeans_distance_reject"]
    rows = []
    for ds in DATASETS:
        for method in methods:
            sub = [r for r in main if r["dataset"] == ds and r["method"] == method]
            if not sub:
                continue
            acc = np.mean([float(r["acc_kept"]) for r in sub])
            rej = np.mean([float(r["rejection_rate"]) for r in sub])
            rows.append({
                "dataset": ds,
                "method": method,
                "coverage": 1 - rej,
                "selective_risk": 1 - acc,
                "acc_kept": acc,
                "rejection_rate": rej,
            })
    write_csv(out / "risk_coverage" / "risk_coverage_points.csv", rows)


def environment_report(out: Path, feature_root: Path) -> None:
    rows = []
    def cmd(command: list[str]) -> str:
        try:
            return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=10).strip()
        except Exception as exc:
            return f"NA: {exc}"
    rows.append({"key": "python", "value": sys.version.replace("\n", " ")})
    rows.append({"key": "platform", "value": platform.platform()})
    rows.append({"key": "cpu", "value": cmd(["bash", "-lc", "lscpu | grep 'Model name' | sed 's/Model name:[[:space:]]*//'"])})
    rows.append({"key": "memory", "value": cmd(["bash", "-lc", "free -h | grep Mem | awk '{print $2}'"])})
    rows.append({"key": "gpu", "value": cmd(["bash", "-lc", "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -5"])})
    for mod in ["numpy", "sklearn", "umap", "torch", "PIL"]:
        try:
            module = __import__(mod)
            rows.append({"key": f"{mod}_version", "value": getattr(module, "__version__", "unknown")})
        except Exception as exc:
            rows.append({"key": f"{mod}_version", "value": f"NA: {exc}"})
    for p in sorted((feature_root / "features").glob("*.npy")):
        rows.append({"key": f"feature_file_{p.name}_bytes", "value": str(p.stat().st_size)})
    write_csv(out / "runtime" / "environment_and_files.csv", rows)


def fit_m_new_full(data_root: Path, feature_root: Path, out: Path, checkpoint: Path, batch_size: int = 32) -> None:
    """Extract M_new features if needed and run exact-dedup main configs."""
    # Use the same extractor as the previous audit script if feature cache does not exist.
    feat_dir = feature_root / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    feat_path = feat_dir / "M_new.npy"
    names_path = feat_dir / "M_new.names.json"
    if not feat_path.exists() or not names_path.exists():
        sys.path.insert(0, str(Path("/data1/D/hostal")))
        from audit_unsupervised_experiments import extract_features
        log("extracting ConvNeXt features for M_new full")
        extract_features(data_root / "M_new", feat_path, checkpoint, batch_size)
    X, names = load_features(feature_root, "M_new")
    # Exact duplicate removal for M_new full: keep first filename per feature-name SHA not available here,
    # so use the file-level exact duplicate audit from scratch.
    import hashlib
    by_sha = {}
    for p in image_files(data_root / "M_new"):
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        by_sha.setdefault(h, []).append(p.name)
    remove = set()
    exact_rows = []
    gid = 0
    for sha, group in sorted(by_sha.items()):
        if len(group) > 1:
            gid += 1
            keep = sorted(group)[0]
            for fname in sorted(group)[1:]:
                remove.add(fname)
                exact_rows.append({"dataset": "M_new", "duplicate_group_id": gid, "kept_file": keep, "removed_file": fname, "sha256": sha})
    idx = [i for i, n in enumerate(names) if n not in remove]
    X = X[idx]
    names = [names[i] for i in idx]
    y = labels_from_names(names)
    write_csv(out / "m_new_full" / "M_new_exact_duplicate_groups.csv", exact_rows)
    rows = []
    for seed in SEEDS:
        log(f"M_new full UMAP100 K20 seed {seed}")
        import umap
        Z = umap.UMAP(n_components=100, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=seed).fit_transform(np.asarray(X))
        km_model = KMeans(n_clusters=20, n_init=10, random_state=seed).fit(Z)
        raw = {
            "kmeans": km_model.labels_,
            "birch": Birch(threshold=0.11, branching_factor=25, n_clusters=20).fit_predict(Z),
            "agg": AgglomerativeClustering(n_clusters=20, linkage="ward").fit_predict(Z),
        }
        aligned = {
            "kmeans": raw["kmeans"],
            "birch": align_to_reference(raw["kmeans"], raw["birch"], 20),
            "agg": align_to_reference(raw["kmeans"], raw["agg"], 20),
        }
        votes = np.vstack([aligned["kmeans"], aligned["birch"], aligned["agg"]])
        all3 = np.all(votes == votes[0:1], axis=0)
        any2 = ((votes[0] == votes[1]).astype(int) + (votes[0] == votes[2]).astype(int) + (votes[1] == votes[2]).astype(int)) > 0
        meta = {"dataset": "M_new", "subset": "exact_dedup_clean", "reduction": "umap", "dim": 100, "k": 20, "seed": seed}
        rows.append(score_external(y, aligned["kmeans"], all3, "all3", meta))
        rows.append(score_external(y, aligned["kmeans"], any2, "any2", meta))
        rows.append(score_external(y, raw["kmeans"], np.ones(len(y), dtype=bool), "kmeans", meta))
        n_rej = int((~all3).sum())
        dist = np.min(((Z[:, None, :] - km_model.cluster_centers_[None, :, :]) ** 2).sum(axis=2), axis=1)
        dkeep = np.ones(len(y), dtype=bool)
        if n_rej:
            dkeep[np.argsort(dist)[-n_rej:]] = False
        rows.append(score_external(y, aligned["kmeans"], dkeep, "kmeans_distance_reject", meta))
    write_csv(out / "m_new_full" / "M_new_full_main.csv", rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("/data1/D"))
    ap.add_argument("--feature-root", type=Path, default=Path("/data1/D/hostal/audit_unsupervised_20260720"))
    ap.add_argument("--followup-root", type=Path, default=Path("/data1/D/hostal/protocolA_followup_20260720"))
    ap.add_argument("--out", type=Path, default=Path("/data1/D/hostal/protocolA_final_supplements_20260720"))
    ap.add_argument("--checkpoint", type=Path, default=Path("/data1/D/deploy/cn/model.safetensors"))
    ap.add_argument("--stage", choices=["audit_assets", "analysis", "m_full", "all"], default="all")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage in {"audit_assets", "all"}:
        build_phash_audit(args.data_root, args.followup_root, args.out)
    if args.stage in {"analysis", "all"}:
        class_level_and_matrices(args.feature_root, args.followup_root, args.out)
        bootstrap_ci(args.followup_root, args.out)
        risk_coverage(args.followup_root, args.out)
        environment_report(args.out, args.feature_root)
    if args.stage in {"m_full", "all"}:
        fit_m_new_full(args.data_root, args.feature_root, args.out, args.checkpoint)
    log(f"completed: {args.out}")


if __name__ == "__main__":
    main()
