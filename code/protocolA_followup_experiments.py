#!/usr/bin/env python3
"""Follow-up Protocol A experiments for the plant-leaf clustering study.

This script intentionally keeps the evaluation protocol as post-hoc clustering
evaluation: disease labels are not used by reduction, clustering, consensus, or
rejection. Labels are only used after clustering to map retained clusters to
classes and compute external metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fftpack import dct
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, Birch, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)


DATASETS = ["F_new", "V_new", "M_new_drop5_drop7", "G_new"]
SEEDS = [11, 22, 33, 44, 55]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def log(msg: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg, flush=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k, v in r.items() if not isinstance(v, (dict, list, np.ndarray))})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def image_files(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def load_feature_set(feature_root: Path, dataset: str) -> tuple[np.ndarray, list[str]]:
    X = np.load(feature_root / "features" / f"{dataset}.npy", mmap_mode="r")
    names = json.loads((feature_root / "features" / f"{dataset}.names.json").read_text(encoding="utf-8"))
    if len(X) != len(names):
        raise ValueError(f"{dataset}: feature rows {len(X)} != names {len(names)}")
    return X, names


def labels_from_names(names: list[str]) -> np.ndarray:
    return np.asarray([int(Path(n).name.split("_", 1)[0]) for n in names], dtype=np.int64)


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def image_pixels(path: Path) -> int:
    try:
        with Image.open(path) as im:
            return int(im.size[0] * im.size[1])
    except Exception:
        return -1


def phash64(path: Path) -> int | None:
    try:
        with Image.open(path) as im:
            arr = np.asarray(im.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    except Exception as exc:
        log(f"pHash skipped {path}: {exc}")
        return None
    coeff = dct(dct(arr, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    flat = coeff.flatten()
    med = np.median(flat[1:])
    bits = flat > med
    out = 0
    for b in bits:
        out = (out << 1) | int(bool(b))
    return out


class BKNode:
    def __init__(self, value: int, item: str):
        self.value = value
        self.items = [item]
        self.children: dict[int, BKNode] = {}


class BKTree:
    def __init__(self):
        self.root: BKNode | None = None

    @staticmethod
    def dist(a: int, b: int) -> int:
        return bin(int(a) ^ int(b)).count("1")

    def add(self, value: int, item: str) -> None:
        if self.root is None:
            self.root = BKNode(value, item)
            return
        node = self.root
        while True:
            d = self.dist(value, node.value)
            if d == 0:
                node.items.append(item)
                return
            child = node.children.get(d)
            if child is None:
                node.children[d] = BKNode(value, item)
                return
            node = child

    def query(self, value: int, radius: int) -> list[tuple[int, str]]:
        if self.root is None:
            return []
        found: list[tuple[int, str]] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            d = self.dist(value, node.value)
            if d <= radius:
                found.extend((d, item) for item in node.items)
            low, high = d - radius, d + radius
            for edge, child in node.children.items():
                if low <= edge <= high:
                    stack.append(child)
        return found


def run_duplicate_audit(data_root: Path, feature_root: Path, out: Path, phash_radius: int) -> None:
    audit = out / "duplicate_audit"
    audit.mkdir(parents=True, exist_ok=True)
    exact_rows: list[dict] = []
    near_rows: list[dict] = []
    keep_map: dict[str, list[str]] = {}

    for ds in DATASETS:
        log(f"duplicate audit: {ds}")
        paths = image_files(data_root / ds)
        by_sha: dict[str, list[Path]] = {}
        for p in paths:
            by_sha.setdefault(sha256_file(p), []).append(p)
        removed: set[str] = set()
        group_id = 0
        for sha, group in sorted(by_sha.items()):
            if len(group) <= 1:
                continue
            group_id += 1
            ranked = sorted(group, key=lambda p: (-image_pixels(p), p.name))
            kept = ranked[0]
            for p in ranked[1:]:
                removed.add(p.name)
                exact_rows.append({
                    "dataset": ds,
                    "duplicate_group_id": group_id,
                    "kept_file": kept.name,
                    "removed_file": p.name,
                    "sha256": sha,
                    "kept_pixels": image_pixels(kept),
                    "removed_pixels": image_pixels(p),
                })
        _, names = load_feature_set(feature_root, ds)
        keep_names = [n for n in names if Path(n).name not in removed]
        keep_map[ds] = keep_names

        tree = BKTree()
        for i, p in enumerate(paths):
            hv = phash64(p)
            if hv is None:
                continue
            for d, other in tree.query(hv, phash_radius):
                near_rows.append({
                    "dataset": ds,
                    "file_a": other,
                    "file_b": p.name,
                    "phash_hamming": d,
                    "highly_suspicious": int(d <= 4),
                })
            tree.add(hv, p.name)
            if (i + 1) % 5000 == 0:
                log(f"pHash {ds}: {i + 1}/{len(paths)}")
        log(f"{ds}: exact duplicate groups={group_id}, removed={len(removed)}, pHash pairs<={phash_radius}={sum(1 for r in near_rows if r['dataset'] == ds)}")

    write_csv(audit / "exact_duplicate_groups.csv", exact_rows)
    write_csv(audit / "near_duplicate_pairs.csv", near_rows)
    (audit / "exact_clean_keep_names.json").write_text(json.dumps(keep_map, ensure_ascii=False), encoding="utf-8")
    log(f"duplicate audit completed: {audit}")


def align_to_reference(ref: np.ndarray, other: np.ndarray, k: int) -> tuple[np.ndarray, dict[int, int]]:
    overlap = np.zeros((k, k), dtype=np.int64)
    for a, b in zip(ref, other):
        if 0 <= a < k and 0 <= b < k:
            overlap[int(a), int(b)] += 1
    rows, cols = linear_sum_assignment(-overlap)
    mapping = {int(c): int(r) for r, c in zip(rows, cols)}
    aligned = np.asarray([mapping.get(int(x), -1) for x in other], dtype=np.int64)
    return aligned, mapping


def reduce_data(X: np.ndarray, method: str, dim: int, seed: int, cache_path: Path) -> np.ndarray:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return np.load(cache_path, mmap_mode="r")
    log(f"reduce: {cache_path.name}")
    if method == "raw":
        Z = np.asarray(X, dtype=np.float32)
    elif method == "pca":
        Z = PCA(n_components=dim, random_state=seed).fit_transform(np.asarray(X))
    elif method == "umap":
        import umap

        Z = umap.UMAP(n_components=dim, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=seed).fit_transform(np.asarray(X))
    else:
        raise ValueError(method)
    np.save(cache_path, np.asarray(Z, dtype=np.float32))
    return np.load(cache_path, mmap_mode="r")


def fit_all(Z: np.ndarray, k: int, seed: int, cache_path: Path) -> dict[str, np.ndarray]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        obj = np.load(cache_path)
        return {name: obj[name] for name in obj.files}
    log(f"cluster: {cache_path.name}")
    Z_arr = np.asarray(Z)
    km_model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z_arr)
    km = km_model.labels_
    birch = Birch(threshold=0.11, branching_factor=25, n_clusters=k).fit_predict(Z_arr)
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z_arr)
    np.savez_compressed(cache_path, kmeans=km.astype(np.int64), birch=birch.astype(np.int64), agg=agg.astype(np.int64))
    return {"kmeans": km, "birch": birch, "agg": agg}


def cluster_votes(raw: dict[str, np.ndarray], k: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    aligned = {"kmeans": np.asarray(raw["kmeans"], dtype=np.int64)}
    aligned["birch"], _ = align_to_reference(aligned["kmeans"], np.asarray(raw["birch"], dtype=np.int64), k)
    aligned["agg"], _ = align_to_reference(aligned["kmeans"], np.asarray(raw["agg"], dtype=np.int64), k)
    votes = np.vstack([aligned["kmeans"], aligned["birch"], aligned["agg"]])
    return aligned, votes


def posthoc_predict(y: np.ndarray, clusters: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    mapping: dict[int, int] = {}
    for c in sorted(set(clusters[keep].tolist())):
        m = keep & (clusters == c)
        if np.any(m):
            mapping[int(c)] = int(Counter(y[m].tolist()).most_common(1)[0][0])
    pred = np.asarray([mapping.get(int(c), -1) for c in clusters], dtype=np.int64)
    return pred, mapping


def external_scores(y: np.ndarray, clusters: np.ndarray, keep: np.ndarray, method: str, meta: dict) -> dict:
    keep = keep & (clusters >= 0)
    pred, mapping = posthoc_predict(y, clusters, keep)
    n = len(y)
    used = int(keep.sum())
    rejected = int(n - used)
    correct = int(np.sum(pred[keep] == y[keep])) if used else 0
    rejected_correct = int(np.sum(pred[~keep] == y[~keep])) if rejected else 0
    kept_error = 1.0 - correct / used if used else math.nan
    rejected_error = 1.0 - rejected_correct / rejected if rejected else math.nan
    row = {
        **meta,
        "method": method,
        "n": n,
        "kept": used,
        "rejected": rejected,
        "rejection_rate": rejected / n if n else math.nan,
        "acc_kept": correct / used if used else math.nan,
        "overall_accuracy": correct / n if n else math.nan,
        "kept_error": kept_error,
        "rejected_error": rejected_error,
        "error_gap": rejected_error - kept_error if np.isfinite(kept_error) and np.isfinite(rejected_error) else math.nan,
        "error_lift": rejected_error / kept_error if kept_error and np.isfinite(kept_error) and np.isfinite(rejected_error) else math.nan,
        "ari_all": adjusted_rand_score(y, clusters),
        "nmi_all": normalized_mutual_info_score(y, clusters),
        "ami_all": adjusted_mutual_info_score(y, clusters),
        "ari_kept": adjusted_rand_score(y[keep], clusters[keep]) if used > 1 else math.nan,
        "nmi_kept": normalized_mutual_info_score(y[keep], clusters[keep]) if used > 1 else math.nan,
        "ami_kept": adjusted_mutual_info_score(y[keep], clusters[keep]) if used > 1 else math.nan,
        "homogeneity_all": homogeneity_score(y, clusters),
        "completeness_all": completeness_score(y, clusters),
        "v_measure_all": v_measure_score(y, clusters),
        "homogeneity_kept": homogeneity_score(y[keep], clusters[keep]) if used > 1 else math.nan,
        "completeness_kept": completeness_score(y[keep], clusters[keep]) if used > 1 else math.nan,
        "v_measure_kept": v_measure_score(y[keep], clusters[keep]) if used > 1 else math.nan,
        "cluster_to_class": mapping,
    }
    return row


def internal_scores(Z: np.ndarray, labels: np.ndarray, meta: dict) -> dict:
    labels = np.asarray(labels)
    n_labels = len(set(labels.tolist()))
    if n_labels < 2 or n_labels >= len(labels):
        return {**meta, "silhouette": math.nan, "davies_bouldin": math.nan, "calinski_harabasz": math.nan}
    Z_arr = np.asarray(Z)
    sample_size = min(3000, len(labels))
    try:
        sil = silhouette_score(Z_arr, labels, sample_size=sample_size, random_state=20260720)
    except Exception:
        sil = math.nan
    try:
        dbi = davies_bouldin_score(Z_arr, labels)
    except Exception:
        dbi = math.nan
    try:
        chi = calinski_harabasz_score(Z_arr, labels)
    except Exception:
        chi = math.nan
    return {**meta, "silhouette": sil, "davies_bouldin": dbi, "calinski_harabasz": chi}


def dataset_clean_view(data_root: Path, feature_root: Path, out: Path, dataset: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X, names = load_feature_set(feature_root, dataset)
    keep_json = out / "duplicate_audit" / "exact_clean_keep_names.json"
    if keep_json.exists():
        keep_names = set(json.loads(keep_json.read_text(encoding="utf-8"))[dataset])
        idx = [i for i, n in enumerate(names) if n in keep_names]
        names = [names[i] for i in idx]
        X = X[idx]
    y = labels_from_names(names)
    return X, y, names


def run_config(
    X: np.ndarray,
    y: np.ndarray,
    out: Path,
    dataset: str,
    subset: str,
    reduction: str,
    dim: int,
    k: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    emb_name = f"{dataset}__{subset}__{reduction}{dim}__seed{seed}.npy"
    Z = reduce_data(X, reduction, dim, seed, out / "cache" / "embeddings" / emb_name)
    cl_name = f"{dataset}__{subset}__{reduction}{dim}__k{k}__seed{seed}.npz"
    raw = fit_all(Z, k, seed, out / "cache" / "clusters" / cl_name)
    aligned, votes = cluster_votes(raw, k)
    return Z, raw, {"aligned": aligned, "votes": votes}


def evaluate_base_methods(
    Z: np.ndarray,
    raw: dict[str, np.ndarray],
    aligned: dict[str, np.ndarray],
    votes: np.ndarray,
    y: np.ndarray,
    meta: dict,
    random_repeats: int = 20,
) -> tuple[list[dict], list[dict], list[dict]]:
    rows: list[dict] = []
    random_rows: list[dict] = []
    internal_rows: list[dict] = []

    all3 = np.all(votes == votes[0:1], axis=0)
    any2 = ((votes[0] == votes[1]).astype(int) + (votes[0] == votes[2]).astype(int) + (votes[1] == votes[2]).astype(int)) > 0
    method_defs = [
        ("all3", aligned["kmeans"], all3),
        ("any2", aligned["kmeans"], any2),
        ("kmeans", raw["kmeans"], np.ones(len(y), dtype=bool)),
        ("birch", raw["birch"], np.ones(len(y), dtype=bool)),
        ("agg", raw["agg"], np.ones(len(y), dtype=bool)),
    ]
    for method, clusters, keep in method_defs:
        rows.append(external_scores(y, np.asarray(clusters), keep, method, meta))
        internal_rows.append(internal_scores(Z, np.asarray(clusters), {**meta, "method": method}))

    kept_n = int(all3.sum())
    rng = np.random.RandomState(int(meta["seed"]) + 20260720)
    for repeat in range(random_repeats):
        keep = np.zeros(len(y), dtype=bool)
        keep[rng.choice(len(y), kept_n, replace=False)] = True
        random_rows.append(external_scores(y, aligned["kmeans"], keep, "random_reject_same_coverage", {**meta, "repeat": repeat}))

    km_model = KMeans(n_clusters=int(meta["k"]), n_init=10, random_state=int(meta["seed"])).fit(np.asarray(Z))
    dist = np.min(((np.asarray(Z)[:, None, :] - km_model.cluster_centers_[None, :, :]) ** 2).sum(axis=2), axis=1)
    keep = np.ones(len(y), dtype=bool)
    n_rej = int((~all3).sum())
    if n_rej:
        keep[np.argsort(dist)[-n_rej:]] = False
    rows.append(external_scores(y, aligned["kmeans"], keep, "kmeans_distance_reject", meta))
    return rows, random_rows, internal_rows


def make_fixed_subset(X: np.ndarray, y: np.ndarray, dataset: str, n: int) -> tuple[np.ndarray, np.ndarray, str]:
    if len(y) <= n:
        return X, y, "full_clean"
    rng = np.random.RandomState(20260720 + len(y))
    idx = np.sort(rng.choice(len(y), n, replace=False))
    return X[idx], y[idx], f"fixed_{n}"


def run_followup(data_root: Path, feature_root: Path, out: Path, subset_n: int) -> None:
    result_dir = out / "followup_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    base_rows: list[dict] = []
    random_rows: list[dict] = []
    fair_rows: list[dict] = []
    trend_rows: list[dict] = []
    param5_rows: list[dict] = []
    internal_rows: list[dict] = []
    stability_rows: list[dict] = []

    param_trend_dims = [50, 100, 200]
    param_trend_ks = [4, 6, 8, 10, 12, 15, 20, 25, 30]
    param5 = [(50, 10), (50, 20), (100, 15), (100, 20), (100, 30), (200, 15), (200, 20)]

    for ds in DATASETS:
        X_full, y_full, _ = dataset_clean_view(data_root, feature_root, out, ds)
        log(f"dataset clean view: {ds}, n={len(y_full)}, classes={len(set(y_full.tolist()))}")

        for seed in SEEDS:
            meta = {"dataset": ds, "subset": "exact_dedup_clean", "reduction": "umap", "dim": 100, "k": 20, "seed": seed}
            Z, raw, packed = run_config(X_full, y_full, out, ds, "exact_dedup_clean", "umap", 100, 20, seed)
            rows, rnd, ints = evaluate_base_methods(Z, raw, packed["aligned"], packed["votes"], y_full, meta)
            base_rows.extend(rows)
            random_rows.extend(rnd)
            internal_rows.extend(ints)
            write_csv(result_dir / "clean_main_incremental.csv", base_rows)
            write_csv(result_dir / "random_same_coverage_incremental.csv", random_rows)

        X_sub, y_sub, subset_tag = make_fixed_subset(X_full, y_full, ds, subset_n)
        for seed in SEEDS:
            for reduction, dim in [("raw", 2048), ("pca", 100), ("umap", 100)]:
                meta = {"dataset": ds, "subset": subset_tag, "reduction": reduction, "dim": dim, "k": 20, "seed": seed}
                Z, raw, packed = run_config(X_sub, y_sub, out, ds, subset_tag, reduction, dim, 20, seed)
                keep = np.all(packed["votes"] == packed["votes"][0:1], axis=0)
                fair_rows.append(external_scores(y_sub, packed["aligned"]["kmeans"], keep, "all3", meta))
                internal_rows.append(internal_scores(Z, packed["aligned"]["kmeans"], {**meta, "method": "all3"}))
                write_csv(result_dir / "fair_reduction_5seeds_incremental.csv", fair_rows)

        for dim in param_trend_dims:
            for k in param_trend_ks:
                seed = 11
                meta = {"dataset": ds, "subset": "exact_dedup_clean", "reduction": "umap", "dim": dim, "k": k, "seed": seed}
                Z, raw, packed = run_config(X_full, y_full, out, ds, "exact_dedup_clean", "umap", dim, k, seed)
                keep = np.all(packed["votes"] == packed["votes"][0:1], axis=0)
                trend_rows.append(external_scores(y_full, packed["aligned"]["kmeans"], keep, "all3", meta))
                internal_rows.append(internal_scores(Z, packed["aligned"]["kmeans"], {**meta, "method": "all3"}))
                write_csv(result_dir / "parameter_trend_smallK_incremental.csv", trend_rows)

        for dim, k in param5:
            labels_by_seed: list[np.ndarray] = []
            for seed in SEEDS:
                meta = {"dataset": ds, "subset": "exact_dedup_clean", "reduction": "umap", "dim": dim, "k": k, "seed": seed}
                Z, raw, packed = run_config(X_full, y_full, out, ds, "exact_dedup_clean", "umap", dim, k, seed)
                keep = np.all(packed["votes"] == packed["votes"][0:1], axis=0)
                param5_rows.append(external_scores(y_full, packed["aligned"]["kmeans"], keep, "all3", meta))
                internal_rows.append(internal_scores(Z, packed["aligned"]["kmeans"], {**meta, "method": "all3"}))
                labels_by_seed.append(np.asarray(packed["aligned"]["kmeans"]))
                write_csv(result_dir / "parameter_sensitivity_5seeds_incremental.csv", param5_rows)
            pair_scores = []
            for i in range(len(labels_by_seed)):
                for j in range(i + 1, len(labels_by_seed)):
                    pair_scores.append(adjusted_rand_score(labels_by_seed[i], labels_by_seed[j]))
            stability_rows.append({
                "dataset": ds,
                "subset": "exact_dedup_clean",
                "reduction": "umap",
                "dim": dim,
                "k": k,
                "seed_count": len(SEEDS),
                "cross_seed_cluster_ari_mean": float(np.mean(pair_scores)),
                "cross_seed_cluster_ari_std": float(np.std(pair_scores)),
            })
            write_csv(result_dir / "cross_seed_stability_incremental.csv", stability_rows)

    write_csv(result_dir / "clean_main.csv", base_rows)
    write_csv(result_dir / "random_same_coverage_completed.csv", random_rows)
    write_csv(result_dir / "fair_reduction_5seeds.csv", fair_rows)
    write_csv(result_dir / "parameter_trend_smallK.csv", trend_rows)
    write_csv(result_dir / "parameter_sensitivity_5seeds.csv", param5_rows)
    write_csv(result_dir / "internal_metrics.csv", internal_rows)
    write_csv(result_dir / "cross_seed_stability.csv", stability_rows)
    log(f"follow-up experiments completed: {result_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("/data1/D"))
    ap.add_argument("--feature-root", type=Path, default=Path("/data1/D/hostal/audit_unsupervised_20260720"))
    ap.add_argument("--out", type=Path, default=Path("/data1/D/hostal/protocolA_followup_20260720"))
    ap.add_argument("--subset-n", type=int, default=6000)
    ap.add_argument("--phash-radius", type=int, default=8)
    ap.add_argument("--stage", choices=["audit", "experiments", "all"], default="all")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(20260720)
    np.random.seed(20260720)
    if args.stage in {"audit", "all"}:
        run_duplicate_audit(args.data_root, args.feature_root, args.out, args.phash_radius)
    if args.stage in {"experiments", "all"}:
        run_followup(args.data_root, args.feature_root, args.out, args.subset_n)


if __name__ == "__main__":
    main()
