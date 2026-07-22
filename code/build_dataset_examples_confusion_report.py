#!/usr/bin/env python3
"""Build a dataset-example and confusion-matrix web report from real files.

This script copies representative original images from the SHA256-cleaned
dataset folders and renders archived real confusion-matrix CSVs. It never
infers matrices from manuscript summary numbers.
"""
from __future__ import annotations

import csv
import html
import json
import shutil
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FULL_ARCHIVE = REPO_ROOT.parent / "github_repro_archive_plant_disease_selective_clustering_20260721"
OUT = REPO_ROOT / "reports" / "dataset_examples_confusion_report_20260722"

IMAGE_ROOT = FULL_ARCHIVE / "data" / "images_clean"
MATRIX_ROOT = FULL_ARCHIVE / "reports" / "protocolA_final_submission_v3" / "class_level_5seed_matrices"
RESULTS_ROOT = FULL_ARCHIVE / "results"

DATASETS = {
    "PV-Fruit": {
        "folder": "F_new",
        "labels": {
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
        "example_labels": [0, 1, 2, 6, 8],
    },
    "PV-Vegetable": {
        "folder": "V_new",
        "labels": {
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
        "example_labels": [0, 2, 4, 5, 11],
    },
    "MCLD-11": {
        "folder": "M_new",
        "labels": {
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
        "example_labels": [0, 3, 5, 8, 10],
    },
    "DFLD-BR-4": {
        "folder": "G_new",
        "labels": {
            0: "Gourd",
            1: "Hibiscus",
            2: "Papaya",
            3: "Zucchini",
        },
        "example_labels": [0, 1, 2, 3],
    },
}

MATRICES = {
    "PV-Vegetable kept": MATRIX_ROOT / "V_new_5seed_avg_kept_confusion_row_normalized.csv",
    "PV-Vegetable all + Rejected": MATRIX_ROOT / "V_new_5seed_avg_all_confusion_with_rejected_row_normalized.csv",
    "MCLD-9 kept": MATRIX_ROOT / "M_new_drop5_drop7_5seed_avg_kept_confusion_row_normalized.csv",
    "MCLD-9 all + Rejected": MATRIX_ROOT / "M_new_drop5_drop7_5seed_avg_all_confusion_with_rejected_row_normalized.csv",
    "PV-Fruit kept": MATRIX_ROOT / "F_new_5seed_avg_kept_confusion_row_normalized.csv",
    "PV-Fruit all + Rejected": MATRIX_ROOT / "F_new_5seed_avg_all_confusion_with_rejected_row_normalized.csv",
    "DFLD-BR-4 kept": MATRIX_ROOT / "G_new_5seed_avg_kept_confusion_row_normalized.csv",
    "DFLD-BR-4 all + Rejected": MATRIX_ROOT / "G_new_5seed_avg_all_confusion_with_rejected_row_normalized.csv",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def first_image_for_label(dataset_folder: str, label: int) -> Path:
    root = IMAGE_ROOT / dataset_folder
    matches = sorted(root.glob(f"{label}_*.jpg"))
    if not matches:
        matches = sorted(root.glob(f"{label}_*.*"))
    if not matches:
        raise FileNotFoundError(f"No image found for {dataset_folder} label {label}")
    return matches[0]


def copy_examples() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    examples_root = OUT / "dataset_examples"
    for display_name, meta in DATASETS.items():
        dst_dir = examples_root / display_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for label in meta["example_labels"]:
            src = first_image_for_label(meta["folder"], label)
            label_name = meta["labels"][label]
            safe_label = label_name.replace("/", "-").replace(" ", "_")
            dst_name = f"{label}_{safe_label}__{src.name}"
            dst = dst_dir / dst_name
            shutil.copy2(src, dst)
            rows.append({
                "display_dataset": display_name,
                "source_folder": meta["folder"],
                "class_id": str(label),
                "class_name": label_name,
                "source_file": str(src),
                "copied_file": str(dst.relative_to(OUT)).replace("\\", "/"),
            })
    write_csv(OUT / "dataset_examples_manifest.csv", rows)
    return rows


def copy_matrix_sources() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    raw_root = OUT / "confusion_matrix_results" / "raw_csv"
    for name, src in MATRICES.items():
        dst = raw_root / f"{name.replace(' ', '_').replace('+', 'plus')}.csv"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        rows.append({
            "item": name,
            "source_file": str(src),
            "copied_file": str(dst.relative_to(OUT)).replace("\\", "/"),
            "data_type": "5-seed average row-normalized confusion matrix from archived real CSV",
        })
    extra_sources = [
        RESULTS_ROOT / "class_level_5seed" / "class_level_rejection_5seed_long.csv",
        RESULTS_ROOT / "class_level_5seed" / "class_level_rejection_5seed_summary.csv",
        RESULTS_ROOT / "m_new_full" / "M_new_full_main.csv",
        RESULTS_ROOT / "m_new_full" / "M_new_exact_duplicate_groups.csv",
    ]
    for src in extra_sources:
        dst = raw_root / "supporting" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        rows.append({
            "item": src.stem,
            "source_file": str(src),
            "copied_file": str(dst.relative_to(OUT)).replace("\\", "/"),
            "data_type": "supporting archived real CSV",
        })
    write_csv(OUT / "confusion_matrix_results" / "matrix_source_manifest.csv", rows)
    readme = OUT / "confusion_matrix_results" / "README.md"
    readme.write_text(
        "# Confusion Matrix Result Package\n\n"
        "This package contains archived real confusion-matrix CSVs and supporting seed-level result CSVs.\n\n"
        "Important scope note: local archives contain 5-seed average row-normalized matrices for "
        "PV-Vegetable, MCLD-9, PV-Fruit, and DFLD-BR-4, plus the MCLD-11 seed-level summary table. "
        "They do not contain per-sample `true_label, aligned_pred_label, kept, seed` files. "
        "No per-sample CSV was inferred from manuscript summary numbers.\n",
        encoding="utf-8",
    )
    return rows


def matrix_to_html(path: Path) -> str:
    rows = read_csv(path)
    if not rows:
        return "<p>CSV为空。</p>"
    cols = [c for c in rows[0].keys() if c != "true_class"]
    parts = ['<div class="matrix-scroll"><table class="matrix"><thead><tr><th>true \\ pred</th>']
    parts += [f"<th>{html.escape(c)}</th>" for c in cols]
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append(f"<tr><th>{html.escape(row.get('true_class', ''))}</th>")
        for col in cols:
            try:
                v = float(row[col])
            except Exception:
                v = 0.0
            alpha = min(0.92, 0.08 + 0.78 * max(0.0, min(1.0, v)))
            text = f"{v * 100:.1f}%"
            parts.append(f'<td style="background:rgba(24,112,96,{alpha:.3f})">{text}</td>')
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def summary_table(path: Path, dataset: str, method: str = "all3") -> str:
    rows = [r for r in read_csv(path) if r.get("dataset") == dataset and r.get("method") == method]
    keep_cols = ["seed", "method", "acc_kept", "rejection_rate", "overall_accuracy", "ari_kept", "nmi_kept", "ami_kept"]
    parts = ["<table><thead><tr>"]
    parts += [f"<th>{c}</th>" for c in keep_cols]
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for col in keep_cols:
            val = row.get(col, "")
            try:
                if col != "seed" and val != "":
                    val = f"{float(val):.4f}"
            except Exception:
                pass
            parts.append(f"<td>{html.escape(str(val))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def write_html(example_rows: list[dict[str, str]], matrix_rows: list[dict[str, str]]) -> None:
    galleries = []
    for display_name in DATASETS:
        imgs = [r for r in example_rows if r["display_dataset"] == display_name]
        cards = []
        for r in imgs:
            rel = r["copied_file"]
            cards.append(
                '<figure>'
                f'<img src="{html.escape(rel)}" alt="{html.escape(display_name + " " + r["class_name"])}">'
                f'<figcaption>{html.escape(r["class_name"])}<br><span>{html.escape(Path(r["source_file"]).name)}</span></figcaption>'
                '</figure>'
            )
        galleries.append(f"<section><h3>{html.escape(display_name)}</h3><div class='gallery'>{''.join(cards)}</div></section>")

    main_matrices = [
        ("PV-Vegetable 保留样本矩阵", MATRICES["PV-Vegetable kept"]),
        ("PV-Vegetable 全样本矩阵（含Rejected列）", MATRICES["PV-Vegetable all + Rejected"]),
        ("MCLD-9 保留样本矩阵", MATRICES["MCLD-9 kept"]),
        ("MCLD-9 全样本矩阵（含Rejected列）", MATRICES["MCLD-9 all + Rejected"]),
    ]
    supplement_matrices = [
        ("PV-Fruit 保留样本矩阵", MATRICES["PV-Fruit kept"]),
        ("PV-Fruit 全样本矩阵（含Rejected列）", MATRICES["PV-Fruit all + Rejected"]),
        ("DFLD-BR-4 保留样本矩阵", MATRICES["DFLD-BR-4 kept"]),
        ("DFLD-BR-4 全样本矩阵（含Rejected列）", MATRICES["DFLD-BR-4 all + Rejected"]),
    ]

    main_html = "".join(f"<section><h3>{title}</h3>{matrix_to_html(path)}</section>" for title, path in main_matrices)
    supp_html = "".join(f"<section><h3>{title}</h3>{matrix_to_html(path)}</section>" for title, path in supplement_matrices)
    mcld11_html = summary_table(RESULTS_ROOT / "m_new_full" / "M_new_full_main.csv", "M_new", "all3")

    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>数据集代表图片与混淆矩阵真实结果报告</title>
<style>
body{{margin:0;background:#f6f7f9;color:#18212f;font-family:Arial,'Microsoft YaHei',sans-serif;line-height:1.55}}
header{{background:#163b45;color:white;padding:28px 40px}}
main{{max-width:1180px;margin:0 auto;padding:24px}}
h1{{margin:0 0 8px;font-size:28px}} h2{{margin:28px 0 12px;border-left:5px solid #187060;padding-left:10px}} h3{{margin:18px 0 10px}}
.note{{background:#fff;border:1px solid #dce3e8;border-radius:8px;padding:14px 16px;margin:14px 0}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px}}
figure{{margin:0;background:#fff;border:1px solid #dce3e8;border-radius:8px;overflow:hidden}}
figure img{{width:100%;height:150px;object-fit:cover;display:block;background:#eef2f5}}
figcaption{{padding:9px 10px;font-size:13px}} figcaption span{{color:#667085;font-size:12px}}
.matrix-scroll{{overflow:auto;background:#fff;border:1px solid #dce3e8;border-radius:8px;margin-bottom:18px}}
table{{border-collapse:collapse;width:max-content;min-width:100%;font-size:12px;background:#fff}}
th,td{{border:1px solid #dce3e8;padding:6px 8px;text-align:center;white-space:nowrap}}
th{{background:#eef3f4;color:#1f3138;position:sticky;top:0;z-index:1}}
.matrix th:first-child{{position:sticky;left:0;z-index:2;text-align:left;max-width:220px;white-space:normal}}
.matrix td{{min-width:62px;color:#071b17;font-weight:600}}
a{{color:#0f766e}} code{{background:#eef3f4;padding:2px 5px;border-radius:4px}}
.links{{display:flex;gap:12px;flex-wrap:wrap}} .button{{background:#187060;color:white;text-decoration:none;padding:9px 12px;border-radius:6px}}
</style>
</head>
<body>
<header>
<h1>数据集代表图片与混淆矩阵真实结果报告</h1>
<p>使用真实清洗数据集原图和已归档真实聚类结果CSV生成；未根据论文汇总数字推测或补画。</p>
</header>
<main>
<section class="note">
<h2>数据口径</h2>
<p>代表图片来自完整归档中的 <code>data/images_clean/</code>，为SHA256去重后实验使用的真实图像文件。混淆矩阵来自归档CSV中的5个随机种子平均按行归一化矩阵。当前本地归档未包含逐样本 <code>true_label, aligned_pred_label, kept, seed</code> CSV，因此本报告没有伪造样本级CSV；压缩包中保留真实矩阵CSV和MCLD-11 seed级汇总表。</p>
<div class="links">
<a class="button" href="dataset_examples.zip">下载数据集代表图片包</a>
<a class="button" href="confusion_matrix_results.zip">下载混淆矩阵原始结果包</a>
</div>
</section>
<h2>图2候选：四个数据集的真实代表性叶片图像</h2>
{''.join(galleries)}
<h2>正文混淆矩阵图候选</h2>
{main_html}
<h2>补充材料矩阵</h2>
{supp_html}
<section>
<h3>MCLD-11 seed级all3结果表</h3>
<p>MCLD-11当前本地归档包含5个seed的主结果CSV，但未包含5-seed平均混淆矩阵或逐样本预测CSV。下表直接读取 <code>results/m_new_full/M_new_full_main.csv</code>。</p>
{mcld11_html}
</section>
<section>
<h2>文件清单</h2>
<p>图片包清单：<a href="dataset_examples_manifest.csv">dataset_examples_manifest.csv</a></p>
<p>矩阵包清单：<a href="confusion_matrix_results/matrix_source_manifest.csv">matrix_source_manifest.csv</a></p>
</section>
</main>
</body>
</html>"""
    (OUT / "index.html").write_text(html_doc, encoding="utf-8")


def zip_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src.parent))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    example_rows = copy_examples()
    matrix_rows = copy_matrix_sources()
    write_html(example_rows, matrix_rows)
    zip_dir(OUT / "dataset_examples", OUT / "dataset_examples.zip")
    zip_dir(OUT / "confusion_matrix_results", OUT / "confusion_matrix_results.zip")
    manifest = {
        "output": str(OUT),
        "dataset_examples_zip": str(OUT / "dataset_examples.zip"),
        "confusion_matrix_results_zip": str(OUT / "confusion_matrix_results.zip"),
        "html": str(OUT / "index.html"),
        "source_note": "Real images and archived real CSVs only; no AI-generated images or inferred matrices.",
    }
    (OUT / "build_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
