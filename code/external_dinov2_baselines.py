#!/usr/bin/env python3
"""External DINOv2 baselines for Protocol A.

This script follows the same transductive post-hoc clustering evaluation
protocol as the frozen Protocol A report. Labels are only read after feature
extraction, reduction, clustering, alignment, and rejection are fixed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, Birch, KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


DATASETS = ["F_new", "V_new", "M_new", "M_new_drop5_drop7"]
SEEDS = [11, 22, 33, 44, 55]
K = 20
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def log(msg: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg, flush=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k, v in row.items() if not isinstance(v, (dict, list, np.ndarray))})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def image_files(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def build_keep_names(data_root: Path, audit_json: Path, out_root: Path) -> dict[str, list[str]]:
    existing = json.loads(audit_json.read_text(encoding="utf-8"))
    keep: dict[str, list[str]] = {}
    exact_rows: list[dict] = []
    for ds in DATASETS:
        files = image_files(data_root / ds)
        if ds in existing:
            allowed = set(existing[ds])
            keep[ds] = [p.name for p in files if p.name in allowed]
            continue

        by_sha: dict[str, list[Path]] = {}
        for p in files:
            by_sha.setdefault(sha256_file(p), []).append(p)
        removed: set[str] = set()
        for group_id, (sha, group) in enumerate(sorted(by_sha.items()), 1):
            if len(group) <= 1:
                continue
            ranked = sorted(group, key=lambda p: p.name)
            kept = ranked[0]
            for p in ranked[1:]:
                removed.add(p.name)
                exact_rows.append({
                    "dataset": ds,
                    "duplicate_group_id": group_id,
                    "kept_file": kept.name,
                    "removed_file": p.name,
                    "sha256": sha,
                })
        keep[ds] = [p.name for p in files if p.name not in removed]
    (out_root / "audit").mkdir(parents=True, exist_ok=True)
    (out_root / "audit" / "exact_clean_keep_names_external.json").write_text(
        json.dumps(keep, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(out_root / "audit" / "exact_duplicate_groups_external.csv", exact_rows)
    return keep


def labels_from_names(names: list[str]) -> np.ndarray:
    return np.asarray([int(Path(n).name.split("_", 1)[0]) for n in names], dtype=np.int64)


class ImageNameDataset(Dataset):
    def __init__(self, root: Path, names: list[str], tfm):
        self.root = root
        self.names = names
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        img = Image.open(self.root / name).convert("RGB")
        return self.tfm(img), name


def load_dinov2_model(device: torch.device, checkpoint: Optional[Path] = None):
    import timm

    use_local = checkpoint is not None and checkpoint.exists()
    model = timm.create_model("vit_base_patch14_dinov2", pretrained=not use_local, num_classes=0, img_size=224)
    if use_local:
        from safetensors.torch import load_file
        from timm.layers import resample_abs_pos_embed

        state = load_file(str(checkpoint), device="cpu")
        current = model.state_dict()
        if "pos_embed" in state and state["pos_embed"].shape != current["pos_embed"].shape:
            state["pos_embed"] = resample_abs_pos_embed(
                state["pos_embed"],
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=1,
            )
        missing, unexpected = model.load_state_dict(state, strict=False)
        log(f"DINOv2 local checkpoint loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval().to(device)
    return model


def extract_dinov2_features(data_root: Path, out_root: Path, keep: dict[str, list[str]], batch_size: int, checkpoint: Optional[Path]) -> None:
    feat_dir = out_root / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dinov2_model(device, checkpoint)
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    torch.set_grad_enabled(False)
    for ds in DATASETS:
        out_feat = feat_dir / f"{ds}.npy"
        out_names = feat_dir / f"{ds}.names.json"
        if out_feat.exists() and out_names.exists():
            log(f"features cached: {ds}")
            continue
        names = keep[ds]
        log(f"extract DINOv2 features: {ds} n={len(names)}")
        loader = DataLoader(
            ImageNameDataset(data_root / ds, names, tfm),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        chunks = []
        seen_names = []
        t0 = time.time()
        for i, (x, batch_names) in enumerate(loader, 1):
            x = x.to(device, non_blocking=True)
            with torch.no_grad():
                z = model(x).detach().cpu().numpy().astype("float32")
            chunks.append(z)
            seen_names.extend(list(batch_names))
            if i % 20 == 0:
                log(f"{ds}: batch {i}/{len(loader)}")
        X = np.concatenate(chunks, axis=0)
        np.save(out_feat, X)
        out_names.write_text(json.dumps(seen_names, ensure_ascii=False), encoding="utf-8")
        log(f"{ds}: saved {X.shape} in {time.time() - t0:.1f}s")


def align_to_reference(ref: np.ndarray, other: np.ndarray, k: int) -> np.ndarray:
    overlap = np.zeros((k, k), dtype=np.int64)
    for a, b in zip(ref, other):
        if 0 <= a < k and 0 <= b < k:
            overlap[int(a), int(b)] += 1
    rows, cols = linear_sum_assignment(-overlap)
    mapping = {int(c): int(r) for r, c in zip(rows, cols)}
    return np.asarray([mapping.get(int(x), -1) for x in other], dtype=np.int64)


def map_predict(y: np.ndarray, clusters: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    mapping: dict[int, int] = {}
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
    return {
        **meta,
        "method": method,
        "n": n,
        "kept": used,
        "coverage": used / n if n else math.nan,
        "rejected": rejected,
        "rejection_rate": rejected / n if n else math.nan,
        "acc_kept": correct / used if used else math.nan,
        "conservative_overall_accuracy": correct / n if n else math.nan,
        "ari_all": adjusted_rand_score(y, clusters),
        "nmi_all": normalized_mutual_info_score(y, clusters),
        "ami_all": adjusted_mutual_info_score(y, clusters),
        "homogeneity_all": homogeneity_score(y, clusters),
        "completeness_all": completeness_score(y, clusters),
        "v_measure_all": v_measure_score(y, clusters),
        "ari_kept": adjusted_rand_score(y[keep], clusters[keep]) if used > 1 else math.nan,
        "nmi_kept": normalized_mutual_info_score(y[keep], clusters[keep]) if used > 1 else math.nan,
        "ami_kept": adjusted_mutual_info_score(y[keep], clusters[keep]) if used > 1 else math.nan,
        "homogeneity_kept": homogeneity_score(y[keep], clusters[keep]) if used > 1 else math.nan,
        "completeness_kept": completeness_score(y[keep], clusters[keep]) if used > 1 else math.nan,
        "v_measure_kept": v_measure_score(y[keep], clusters[keep]) if used > 1 else math.nan,
    }


def fit_or_load_clusters(Z: np.ndarray, cache_path: Path, seed: int, k: int = K) -> dict[str, np.ndarray]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        obj = np.load(cache_path)
        return {name: obj[name] for name in obj.files}
    log(f"cluster: {cache_path.name}")
    Z_arr = np.asarray(Z)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z_arr)
    birch = Birch(threshold=0.11, branching_factor=25, n_clusters=k).fit_predict(Z_arr)
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z_arr)
    np.savez_compressed(cache_path, kmeans=km.astype(np.int64), birch=birch.astype(np.int64), agg=agg.astype(np.int64))
    return {"kmeans": km.astype(np.int64), "birch": birch.astype(np.int64), "agg": agg.astype(np.int64)}


def reduce_umap(X: np.ndarray, cache_path: Path, seed: int) -> np.ndarray:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return np.load(cache_path, mmap_mode="r")
    import umap

    log(f"UMAP: {cache_path.name}")
    Z = umap.UMAP(
        n_components=100,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=seed,
    ).fit_transform(np.asarray(X))
    np.save(cache_path, np.asarray(Z, dtype=np.float32))
    return np.load(cache_path, mmap_mode="r")


def run_clustering(out_root: Path) -> None:
    rows: list[dict] = []
    for ds in DATASETS:
        X = np.load(out_root / "features" / f"{ds}.npy", mmap_mode="r")
        names = json.loads((out_root / "features" / f"{ds}.names.json").read_text(encoding="utf-8"))
        y = labels_from_names(names)
        for seed in SEEDS:
            meta = {"dataset": ds, "seed": seed, "k": K, "feature": "DINOv2-vit_base_patch14-224"}

            km_cache = out_root / "cache" / "clusters" / f"{ds}__dinov2_raw__k{K}__seed{seed}.npz"
            raw = fit_or_load_clusters(X, km_cache, seed)
            all_keep = np.ones(len(y), dtype=bool)
            rows.append(score_external(y, raw["kmeans"], all_keep, "DINOv2+KMeans", meta | {"reduction": "raw"}))

            Z = reduce_umap(X, out_root / "cache" / "embeddings" / f"{ds}__dinov2_umap100__seed{seed}.npy", seed)
            clu = fit_or_load_clusters(Z, out_root / "cache" / "clusters" / f"{ds}__dinov2_umap100__k{K}__seed{seed}.npz", seed)
            aligned = {
                "kmeans": clu["kmeans"],
                "birch": align_to_reference(clu["kmeans"], clu["birch"], K),
                "agg": align_to_reference(clu["kmeans"], clu["agg"], K),
            }
            votes = np.vstack([aligned["kmeans"], aligned["birch"], aligned["agg"]])
            any2 = (votes[0] == votes[1]) | (votes[0] == votes[2]) | (votes[1] == votes[2])
            all3 = (votes[0] == votes[1]) & (votes[0] == votes[2])
            rows.append(score_external(y, aligned["kmeans"], all_keep, "DINOv2+UMAP+KMeans", meta | {"reduction": "umap100"}))
            rows.append(score_external(y, aligned["birch"], all_keep, "DINOv2+UMAP+Birch", meta | {"reduction": "umap100"}))
            rows.append(score_external(y, aligned["agg"], all_keep, "DINOv2+UMAP+Agglomerative", meta | {"reduction": "umap100"}))
            rows.append(score_external(y, aligned["kmeans"], any2, "DINOv2+UMAP+any2", meta | {"reduction": "umap100"}))
            rows.append(score_external(y, aligned["kmeans"], all3, "DINOv2+UMAP+all3", meta | {"reduction": "umap100"}))
            write_csv(out_root / "external_dinov2_results.csv", rows)

    write_csv(out_root / "external_dinov2_results.csv", rows)
    summarize(out_root / "external_dinov2_results.csv", out_root / "external_dinov2_summary.csv")


def summarize(result_csv: Path, summary_csv: Path) -> None:
    rows = []
    with result_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["method"]), []).append(row)
    metrics = [
        "coverage",
        "rejection_rate",
        "acc_kept",
        "conservative_overall_accuracy",
        "ari_all",
        "nmi_all",
        "ami_all",
        "homogeneity_all",
        "completeness_all",
        "v_measure_all",
        "ari_kept",
        "nmi_kept",
        "ami_kept",
        "homogeneity_kept",
        "completeness_kept",
        "v_measure_kept",
    ]
    out = []
    for (ds, method), items in sorted(groups.items()):
        row = {"dataset": ds, "method": method, "runs": len(items)}
        for m in metrics:
            vals = np.asarray([float(x[m]) for x in items], dtype=float)
            row[f"{m}_mean"] = float(np.nanmean(vals))
            row[f"{m}_sd"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        out.append(row)
    write_csv(summary_csv, out)


def build_html(out_root: Path) -> None:
    summary = list(csv.DictReader((out_root / "external_dinov2_summary.csv").open(encoding="utf-8")))

    def pct(x: str) -> str:
        return f"{float(x) * 100:.2f}%"

    def num(x: str) -> str:
        return f"{float(x):.3f}"

    rows_html = []
    for r in summary:
        rows_html.append(
            "<tr>"
            f"<td>{r['dataset']}</td><td>{r['method']}</td><td>{r['runs']}</td>"
            f"<td>{pct(r['acc_kept_mean'])} ± {pct(r['acc_kept_sd'])}</td>"
            f"<td>{pct(r['coverage_mean'])} ± {pct(r['coverage_sd'])}</td>"
            f"<td>{pct(r['conservative_overall_accuracy_mean'])} ± {pct(r['conservative_overall_accuracy_sd'])}</td>"
            f"<td>{num(r['ari_all_mean'])}</td><td>{num(r['nmi_all_mean'])}</td><td>{num(r['ami_all_mean'])}</td>"
            f"<td>{num(r['ari_kept_mean'])}</td><td>{num(r['nmi_kept_mean'])}</td><td>{num(r['ami_kept_mean'])}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>DINOv2外部无监督基线补充报告</title>
<style>
body{{font-family:Arial,'Microsoft YaHei',sans-serif;margin:28px;color:#1f2937;line-height:1.55}}
h1,h2{{color:#111827}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0 24px}}
th,td{{border:1px solid #d1d5db;padding:7px 8px;text-align:right}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
th{{background:#f3f4f6}}
.note{{background:#f8fafc;border-left:4px solid #64748b;padding:10px 14px;margin:14px 0}}
code{{background:#f3f4f6;padding:1px 4px;border-radius:3px}}
</style>
</head>
<body>
<h1>DINOv2外部无监督基线补充报告</h1>
<div class="note">
本补充实验采用传导式无监督聚类评价协议。真实标签不参与DINOv2特征提取、UMAP、聚类、Hungarian簇编号对齐或共识拒识，仅在结果固定后用于多对一事后类别对齐和外部聚类评价。
</div>
<h2>一、实验设置</h2>
<p>数据集：F_new、V_new、完整11类M_new、预先整理9类M_new_drop5_drop7。所有数据采用SHA256去重图像；其中M_new在本补充脚本中执行同一SHA256去重流程。</p>
<p>DINOv2模型：<code>timm vit_base_patch14_dinov2</code>公开自监督预训练权重，冻结特征提取器；输入尺寸固定为224×224，并使用ImageNet均值方差归一化。主簇数固定为K=20，随机种子为11、22、33、44、55。</p>
<p>比较方法包括：DINOv2+KMeans全样本基线，以及DINOv2+UMAP100+KMeans/Birch/Agglomerative/any2/all3。注意：全样本结果和选择性拒识结果不能直接按同一准确率含义排名。</p>
<h2>二、结果汇总</h2>
<table>
<thead><tr><th>数据集</th><th>方法</th><th>运行数</th><th>保留样本事后对齐聚类准确率</th><th>Coverage</th><th>保守全样本准确率</th><th>ARI_all</th><th>NMI_all</th><th>AMI_all</th><th>ARI_kept</th><th>NMI_kept</th><th>AMI_kept</th></tr></thead>
<tbody>
{''.join(rows_html)}
</tbody>
</table>
<h2>三、文件</h2>
<p>原始逐种子结果：<code>external_dinov2_results.csv</code>；均值±标准差汇总：<code>external_dinov2_summary.csv</code>；SHA256补充记录：<code>audit/exact_duplicate_groups_external.csv</code>。</p>
<h2>四、SCAN复现状态</h2>
<p>本轮未纳入SCAN数值结果。原因是服务器未发现既有SCAN复现代码，且SCAN需要独立的自监督预训练/邻域挖掘/聚类头训练流程；在没有完整可复现配置的情况下，不将其作为定量表格结果，以避免引入不可核查或不公平的基线。</p>
</body>
</html>
"""
    report = out_root / "web_report_external_dinov2" / "index.html"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(html, encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=repo_root / "data" / "images_clean")
    parser.add_argument("--out", type=Path, default=repo_root / "outputs" / "external_baselines_dinov2")
    parser.add_argument("--audit-json", type=Path, default=repo_root / "data_audit" / "duplicate_audit" / "exact_clean_keep_names.json")
    parser.add_argument("--dinov2-checkpoint", type=Path, default=repo_root / "models" / "dinov2_vit_base_patch14_lvd142m.safetensors")
    parser.add_argument("--batch-size", type=int, default=96)
    args = parser.parse_args()

    data_root = args.data_root
    out_root = args.out
    audit_json = args.audit_json
    keep = build_keep_names(data_root, audit_json, out_root)
    extract_dinov2_features(data_root, out_root, keep, batch_size=args.batch_size, checkpoint=args.dinov2_checkpoint)
    run_clustering(out_root)
    build_html(out_root)
    log(f"done: {out_root}")


if __name__ == "__main__":
    main()
