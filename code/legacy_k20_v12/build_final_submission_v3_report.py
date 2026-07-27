#!/usr/bin/env python3
"""Build the final integrated submission report v3.

This version is lightweight and manuscript-oriented:
- includes all completed experiment sections;
- uses conservative metric names;
- copies small parameter/environment/summary CSV files into the report folder;
- does not copy large image assets.
"""
from __future__ import annotations

import csv
import html
import math
import shutil
from pathlib import Path

import pandas as pd


if Path(r"F:\hospitol").exists():
    ROOT = Path(r"F:\hospitol")
else:
    ROOT = Path("/data1/D/hostal")

BASE = ROOT / "protocolA_followup_20260720"
SUP = ROOT / "protocolA_final_supplements_20260720"
PATCH = ROOT / "protocolA_final_required_patch_20260720"
TIMING = ROOT / "protocolA_timing_vnew_20260720"
FIG_DIR = ROOT / "protocolA_manuscript_figures_20260720"
OUT = BASE / "web_report_final_submission_v3"
OUT.mkdir(parents=True, exist_ok=True)

MAIN_DATASETS = ["F_new", "V_new", "M_new_drop5_drop7"]
ALL_DATASETS = ["F_new", "V_new", "M_new_drop5_drop7", "G_new"]
LABEL = {
    "F_new": "F_new 水果病害子集",
    "V_new": "V_new 蔬菜病害子集",
    "M_new_drop5_drop7": "M_new_drop5_drop7 多作物病害清洗子集",
    "G_new": "G_new 跨作物视觉类别辅助集",
    "M_new": "完整 M_new 多作物病害数据集",
}
METHOD_LABEL = {
    "kmeans": "KMeans",
    "birch": "Birch",
    "agg": "Agglomerative",
    "any2": "any2",
    "all3": "all3",
    "kmeans_distance_reject": "KMeans distance rejection",
    "random_same_coverage": "Random same-coverage rejection",
}


def esc(x) -> str:
    text = str(x).replace("verticulium", "Verticillium").replace("verticillium", "Verticillium")
    return html.escape(text)


def pct(x: float, d: int = 2) -> str:
    if x is None or not math.isfinite(float(x)):
        return "NA"
    return f"{float(x) * 100:.{d}f}%"


def dec(x: float, d: int = 3) -> str:
    if x is None or not math.isfinite(float(x)):
        return "NA"
    return f"{float(x):.{d}f}"


def mean_std_text(s: pd.Series, as_pct: bool = False) -> str:
    m = float(s.mean())
    sd = float(s.std(ddof=0)) if len(s) else math.nan
    return f"{pct(m)} ± {pct(sd)}" if as_pct else f"{dec(m)} ± {dec(sd)}"


def table(headers: list[str], rows: list[list[str]], compact: bool = False) -> str:
    cls = "compact" if compact else ""
    h = "".join(f"<th>{esc(x)}</th>" for x in headers)
    body = "\n".join("<tr>" + "".join(f"<td>{x}</td>" for x in row) + "</tr>" for row in rows)
    return f'<div class="tablebox"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>'


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summary(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: v for c, v in zip(group_cols, keys)}
        row["runs"] = len(sub)
        for m in metrics:
            row[f"{m}_mean"] = sub[m].mean()
            row[f"{m}_std"] = sub[m].std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows)


def copy_small(src: Path, dst_name: str | None = None) -> None:
    if src.exists():
        shutil.copy2(src, OUT / (dst_name or src.name))


def copy_text_sanitized(src: Path, dst: Path) -> None:
    text = src.read_text(encoding="utf-8-sig")
    text = text.replace("verticulium", "Verticillium").replace("verticillium", "Verticillium")
    dst.write_text(text, encoding="utf-8-sig")


def build_main_table(clean: pd.DataFrame) -> str:
    rows = []
    for ds in ALL_DATASETS:
        sub = clean[(clean.dataset == ds) & (clean.method == "all3")]
        rows.append([
            LABEL[ds],
            str(int(sub["n"].iloc[0])),
            mean_std_text(sub["acc_kept"], True),
            mean_std_text(sub["rejection_rate"], True),
            mean_std_text(sub["overall_accuracy"], True),
            mean_std_text(sub["ari_kept"]),
            mean_std_text(sub["nmi_kept"]),
            mean_std_text(sub["ami_kept"]),
        ])
    return table(["数据集", "样本数", "保留样本事后对齐聚类准确率 acc_kept", "拒识率", "保守全样本准确率", "ARI_kept", "NMI_kept", "AMI_kept"], rows)


def build_ablation(clean: pd.DataFrame, random_df: pd.DataFrame) -> str:
    methods = ["kmeans", "birch", "agg", "any2", "all3", "kmeans_distance_reject"]
    rows_df = []
    for ds in MAIN_DATASETS:
        for method in methods:
            sub = clean[(clean.dataset == ds) & (clean.method == method)].copy()
            rows_df.append({
                "dataset": ds,
                "method": METHOD_LABEL[method],
                "rejective": "否" if method in {"kmeans", "birch", "agg"} else "是",
                "acc_kept": mean_std_text(sub["acc_kept"], True),
                "rejection_rate": mean_std_text(sub["rejection_rate"], True),
                "conservative_overall_accuracy": mean_std_text(sub["overall_accuracy"], True),
                "ari_kept": mean_std_text(sub["ari_kept"]),
                "nmi_kept": mean_std_text(sub["nmi_kept"]),
                "ami_kept": mean_std_text(sub["ami_kept"]),
            })
        rnd = random_df[random_df.dataset == ds].groupby("seed", as_index=False).mean(numeric_only=True)
        rows_df.append({
            "dataset": ds,
            "method": METHOD_LABEL["random_same_coverage"],
            "rejective": "是",
            "acc_kept": mean_std_text(rnd["acc_kept"], True),
            "rejection_rate": mean_std_text(rnd["rejection_rate"], True),
            "conservative_overall_accuracy": mean_std_text(rnd["overall_accuracy"], True),
            "ari_kept": mean_std_text(rnd["ari_kept"]),
            "nmi_kept": mean_std_text(rnd["nmi_kept"]),
            "ami_kept": mean_std_text(rnd["ami_kept"]),
        })
    write_csv(OUT / "summary_ablation_complete.csv", rows_df)
    html_rows = [[LABEL[r["dataset"]], r["method"], r["rejective"], r["acc_kept"], r["rejection_rate"], r["conservative_overall_accuracy"], r["ari_kept"], r["nmi_kept"]] for r in rows_df]
    return table(["数据集", "方法", "是否拒识", "acc_kept", "拒识率", "保守全样本准确率", "ARI_kept", "NMI_kept"], html_rows, compact=True)


def build_cluster_metrics(clean: pd.DataFrame) -> str:
    rows = []
    for ds in ALL_DATASETS:
        sub = clean[(clean.dataset == ds) & (clean.method == "all3")]
        rows.append({
            "dataset": ds,
            "ARI_all": mean_std_text(sub["ari_all"]),
            "ARI_kept": mean_std_text(sub["ari_kept"]),
            "NMI_all": mean_std_text(sub["nmi_all"]),
            "NMI_kept": mean_std_text(sub["nmi_kept"]),
            "AMI_all": mean_std_text(sub["ami_all"]),
            "AMI_kept": mean_std_text(sub["ami_kept"]),
            "Homogeneity_all": mean_std_text(sub["homogeneity_all"]),
            "Homogeneity_kept": mean_std_text(sub["homogeneity_kept"]),
            "Completeness_all": mean_std_text(sub["completeness_all"]),
            "Completeness_kept": mean_std_text(sub["completeness_kept"]),
            "V_measure_all": mean_std_text(sub["v_measure_all"]),
            "V_measure_kept": mean_std_text(sub["v_measure_kept"]),
        })
    write_csv(OUT / "summary_all_vs_kept_cluster_metrics.csv", rows)
    html_rows = [[LABEL[r["dataset"]], r["ARI_all"], r["ARI_kept"], r["NMI_all"], r["NMI_kept"], r["AMI_all"], r["AMI_kept"]] for r in rows]
    return table(["数据集", "ARI_all", "ARI_kept", "NMI_all", "NMI_kept", "AMI_all", "AMI_kept"], html_rows)


def build_fair_reduction(fair: pd.DataFrame) -> str:
    fair = fair[fair.dataset.isin(ALL_DATASETS)].copy()
    out = summary(fair, ["dataset", "reduction"], ["acc_kept", "rejection_rate", "overall_accuracy", "ari_kept", "nmi_kept", "ami_kept"])
    order = {"raw": 0, "pca": 1, "umap": 2}
    out["order"] = out["reduction"].map(order)
    out = out.sort_values(["dataset", "order"])
    out.to_csv(OUT / "summary_fair_reduction.csv", index=False, encoding="utf-8-sig")
    rows = []
    for _, r in out.iterrows():
        rows.append([
            LABEL[r.dataset],
            {"raw": "Raw 2048D", "pca": "PCA 100D", "umap": "UMAP 100D"}[r.reduction],
            f"{pct(r.acc_kept_mean)} ± {pct(r.acc_kept_std)}",
            f"{pct(r.rejection_rate_mean)} ± {pct(r.rejection_rate_std)}",
            f"{pct(r.overall_accuracy_mean)} ± {pct(r.overall_accuracy_std)}",
            f"{dec(r.ari_kept_mean)} ± {dec(r.ari_kept_std)}",
            f"{dec(r.nmi_kept_mean)} ± {dec(r.nmi_kept_std)}",
        ])
    return table(["数据集", "表示方式", "acc_kept", "拒识率", "保守全样本准确率", "ARI_kept", "NMI_kept"], rows, compact=True)


def build_parameter_sensitivity(param: pd.DataFrame) -> str:
    param = param[param.dataset.isin(MAIN_DATASETS)].copy()
    out = summary(param, ["dataset", "dim", "k"], ["acc_kept", "rejection_rate", "overall_accuracy", "ari_kept", "nmi_kept", "ami_kept"])
    out = out.sort_values(["dataset", "dim", "k"])
    out.to_csv(OUT / "summary_parameter_sensitivity.csv", index=False, encoding="utf-8-sig")
    rows = []
    for _, r in out.iterrows():
        rows.append([
            LABEL[r.dataset],
            str(int(r.dim)),
            str(int(r.k)),
            f"{pct(r.acc_kept_mean)} ± {pct(r.acc_kept_std)}",
            f"{pct(r.rejection_rate_mean)} ± {pct(r.rejection_rate_std)}",
            f"{pct(r.overall_accuracy_mean)} ± {pct(r.overall_accuracy_std)}",
            f"{dec(r.ari_kept_mean)} ± {dec(r.ari_kept_std)}",
            f"{dec(r.nmi_kept_mean)} ± {dec(r.nmi_kept_std)}",
        ])
    return table(["数据集", "UMAP维度", "K", "acc_kept", "拒识率", "保守全样本准确率", "ARI_kept", "NMI_kept"], rows, compact=True)


def build_bootstrap() -> str:
    ci = pd.read_csv(PATCH / "bootstrap_seedlevel" / "method_difference_seedlevel_bootstrap_ci.csv")
    ci.to_csv(OUT / "summary_seedlevel_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    order = ["all3 - random_same_coverage", "all3 - kmeans_distance_reject", "all3 - any2", "all3 - kmeans"]
    rows = []
    for ds in MAIN_DATASETS:
        for comp in order:
            sub = ci[(ci.dataset == ds) & (ci.comparison == comp) & (ci.metric == "acc_kept")]
            if sub.empty:
                continue
            r = sub.iloc[0]
            rows.append([
                LABEL[ds],
                comp.replace("kmeans", "KMeans"),
                pct(r.mean_diff),
                f"[{pct(r.ci95_low)}, {pct(r.ci95_high)}]",
                "是" if int(r.ci_excludes_zero) else "否",
                "是" if int(r.practically_meaningful) else "否",
            ])
    return table(["数据集", "比较", "acc_kept差值", "种子级95% CI", "CI不跨0", "实际意义≥0.5pp"], rows, compact=True)


def build_class_level() -> str:
    src = PATCH / "class_level_5seed" / "class_level_rejection_5seed_summary.csv"
    weak_src = PATCH / "class_level_5seed" / "weak_or_high_rejection_classes_5seed.csv"
    copy_text_sanitized(src, OUT / src.name)
    copy_text_sanitized(weak_src, OUT / weak_src.name)
    df = pd.read_csv(weak_src)
    rows = []
    for _, r in df[df.dataset.isin(MAIN_DATASETS)].head(12).iterrows():
        rows.append([
            LABEL[r.dataset],
            esc(r.class_name),
            str(int(r.total)),
            f"{float(r.kept_mean):.1f} ± {float(r.kept_std):.1f}",
            f"{pct(r.rejection_rate_mean)} ± {pct(r.rejection_rate_std)}",
            f"{pct(r.acc_kept_mean)} ± {pct(r.acc_kept_std)}",
        ])
    return table(["数据集", "类别", "总数", "保留数", "拒识率", "acc_kept"], rows, compact=True)


def build_risk_coverage() -> str:
    src = PATCH / "risk_coverage_discrete" / "risk_coverage_discrete_operating_points.csv"
    copy_small(src)
    df = pd.read_csv(src)
    rows = []
    for _, r in df[df.dataset.isin(MAIN_DATASETS)].iterrows():
        rows.append([LABEL[r.dataset], METHOD_LABEL.get(r.method, r.method), pct(r.coverage), pct(r.selective_risk), pct(r.acc_kept)])
    return table(["数据集", "策略", "覆盖率", "选择性风险", "acc_kept"], rows, compact=True)


def build_m_compare(clean: pd.DataFrame) -> str:
    full = pd.read_csv(SUP / "m_new_full" / "M_new_full_main.csv")
    rows = []
    for name, df, ds, cls_num in [
        ("完整 M_new", full, "M_new", "11"),
        ("M_new_drop5_drop7 清洗版", clean, "M_new_drop5_drop7", "9"),
    ]:
        sub = df[(df.dataset == ds) & (df.method == "all3")]
        rows.append([
            name,
            cls_num,
            str(int(sub["n"].iloc[0])),
            mean_std_text(sub["acc_kept"], True),
            mean_std_text(sub["rejection_rate"], True),
            mean_std_text(sub["overall_accuracy"], True),
            mean_std_text(sub["ari_kept"]),
            mean_std_text(sub["nmi_kept"]),
            mean_std_text(sub["ami_kept"]),
        ])
    return table(["版本", "类别数", "样本数", "acc_kept", "拒识率", "保守全样本准确率", "ARI_kept", "NMI_kept", "AMI_kept"], rows)


def build_timing() -> str:
    path = TIMING / "timing_vnew_seed11.csv"
    if not path.exists():
        return "<p class='muted'>计时文件暂未生成。</p>"
    copy_small(path)
    metrics = TIMING / "timing_vnew_seed11_metrics.json"
    copy_small(metrics)
    df = pd.read_csv(path)
    keep = [
        "ConvNeXt feature extraction",
        "UMAP 100D fitting and transform",
        "KMeans clustering",
        "Birch clustering",
        "Agglomerative clustering",
        "Hungarian cluster alignment",
        "Consensus and rejection masks",
        "Post-hoc mapping and evaluation",
        "Total measured pipeline",
        "Per-image ConvNeXt feature time",
    ]
    rows = []
    for stage in keep:
        r = df[df.stage == stage]
        if r.empty:
            continue
        sec = float(r.iloc[0].seconds)
        rows.append([stage, f"{sec * 1000:.2f} ms/image" if "Per-image" in stage else f"{sec:.2f} s ({sec / 60:.2f} min)"])
    note = "计时以V_new、seed=11为代表。服务器检测到两张NVIDIA GeForce RTX 4090；PyTorch特征提取使用默认CUDA设备，计时日志中GPU0显存占用约3319 MB。UMAP、KMeans、Birch和Agglomerative使用CPU版umap-learn/scikit-learn实现。"
    return f"<div class='note'>{esc(note)}</div>" + table(["阶段", "耗时"], rows, compact=True)


def build_figure_table() -> str:
    manifest = FIG_DIR / "README_figure_manifest.md"
    if manifest.exists():
        copy_small(manifest, "README_figure_manifest.md")
    rows = [
        ["Fig. 1", "方法总流程", "正文", "无", "PDF/TIFF"],
        ["Fig. 2", "主实验结果", "正文", "clean_main.csv", "PDF/TIFF"],
        ["Fig. 3", "完整消融比较", "正文", "summary_ablation_complete.csv", "PDF/TIFF"],
        ["Fig. 4", "离散风险—覆盖率操作点", "正文", "risk_coverage_discrete_operating_points.csv", "PDF"],
        ["Fig. 5", "参数敏感性热力图", "正文/补充", "summary_parameter_sensitivity.csv", "PDF"],
        ["Fig. 6", "类别级薄弱点", "正文/补充", "weak_or_high_rejection_classes_5seed.csv", "PDF/TIFF"],
        ["Fig. 7", "平均混淆矩阵", "正文", "class_level_5seed/*confusion*.csv", "TIFF"],
        ["Fig. 8", "公平降维对照", "正文/补充", "summary_fair_reduction.csv", "PDF"],
    ]
    note = "完整图件服务器目录：/data1/D/hostal/protocolA_manuscript_figures_20260720。所有图均导出SVG、PDF和TIFF。离散点图不称为AURC曲线；误差线表示5个随机种子的标准差。"
    return f"<div class='note'>{esc(note)}</div>" + table(["图号", "内容", "正文/补充", "数据文件", "推荐格式"], rows)


def prepare_small_assets() -> None:
    copy_small(ROOT / "protocolA_method_parameters.md")
    copy_small(ROOT / "protocolA_M_dataset_exclusion_note.md")
    copy_small(SUP / "runtime" / "environment_and_files.csv")
    matrix_dir = OUT / "class_level_5seed_matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    src_dir = PATCH / "class_level_5seed"
    for p in src_dir.glob("*confusion*.csv"):
        copy_text_sanitized(p, matrix_dir / p.name)


def build_html() -> str:
    prepare_small_assets()
    clean = pd.read_csv(BASE / "followup_results" / "clean_main.csv")
    random_df = pd.read_csv(BASE / "followup_results" / "random_same_coverage_completed.csv")
    fair = pd.read_csv(BASE / "followup_results" / "fair_reduction_5seeds.csv")
    param = pd.read_csv(BASE / "followup_results" / "parameter_sensitivity_5seeds.csv")
    clean.to_csv(OUT / "source_clean_main.csv", index=False, encoding="utf-8-sig")

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Protocol A 最终投稿实验报告 v3</title>
<style>
body{{margin:0;background:#f6f7f9;color:#202a36;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",Arial,sans-serif;line-height:1.65}}
.wrap{{max-width:1200px;margin:0 auto;padding:26px 20px}}header{{background:#fff;border-bottom:1px solid #d8dee8}}
h1{{margin:0 0 8px;font-size:28px}}h2{{font-size:22px;margin:28px 0 12px}}h3{{font-size:17px;margin:18px 0 8px}}
.muted{{color:#667085}}.note{{background:#e7f1ef;border-left:4px solid #256f68;border-radius:0 8px 8px 0;padding:12px 14px;margin:12px 0}}.warn{{background:#f6ead8;border-left-color:#9f5d12}}
.tablebox{{background:#fff;border:1px solid #d8dee8;border-radius:8px;overflow-x:auto;margin:12px 0 20px}}table{{width:100%;border-collapse:collapse;min-width:860px}}th,td{{border-bottom:1px solid #d8dee8;padding:9px 11px;text-align:left;white-space:nowrap;font-size:14px}}th{{background:#eef2f6}}tr:last-child td{{border-bottom:0}}table.compact th,table.compact td{{font-size:12.5px;padding:7px 8px}}
.links{{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 18px}}.links a{{background:#fff;border:1px solid #d8dee8;border-radius:6px;padding:7px 9px;color:#2f5f9e;text-decoration:none;font-size:13px}}code{{background:#edf1f5;padding:2px 5px;border-radius:4px}}
</style>
</head>
<body>
<header><div class="wrap">
<h1>Protocol A 最终投稿实验报告 v3</h1>
<p class="muted">实验层面冻结版。协议为传导式选择性无监督聚类评价：全部图像在不使用真实标签的条件下参与特征空间构建、降维、聚类、簇对齐与拒识；真实标签仅在聚类和拒识结果完全固定后用于多对一事后类别对齐和外部评价。</p>
</div></header>
<main class="wrap">
<section><h2>一、数据审计与指标命名</h2>
<div class="note">SHA256用于删除字节级完全重复图像；pHash仅用于近重复风险审计，不据此删除图像，也不参与特征提取、聚类、拒识或评价。acc_kept表示“保留样本事后对齐聚类准确率”。“保守全样本准确率”定义为 correct_kept / N，等价于 acc_kept × coverage；该指标采用保守口径将拒识样本计入全样本分母，并不表示拒识样本均被错误分类。</div>
</section>
<section><h2>二、主实验结果</h2>{build_main_table(clean)}
<div class="note">UMAP 100维、K=20是跨数据集统一主配置，并非每个数据集上的最优参数。G_new为辅助跨作物视觉类别实验，不进入三个病害主数据集综合平均。</div></section>
<section><h2>三、完整消融与拒识基线</h2>{build_ablation(clean, random_df)}<div class="links"><a href="summary_ablation_complete.csv">summary_ablation_complete.csv</a></div></section>
<section><h2>四、全部样本与保留样本聚类指标</h2>{build_cluster_metrics(clean)}<div class="links"><a href="summary_all_vs_kept_cluster_metrics.csv">summary_all_vs_kept_cluster_metrics.csv</a></div></section>
<section><h2>五、公平降维对照</h2><p class="muted">F_new和V_new使用固定6000张样本；M_new_drop5_drop7与G_new使用全量样本。三种表示均采用K=20和all3规则。</p>{build_fair_reduction(fair)}<div class="links"><a href="summary_fair_reduction.csv">summary_fair_reduction.csv</a></div></section>
<section><h2>六、参数敏感性</h2>{build_parameter_sensitivity(param)}<div class="links"><a href="summary_parameter_sensitivity.csv">summary_parameter_sensitivity.csv</a></div></section>
<section><h2>七、种子级配对Bootstrap</h2>{build_bootstrap()}<div class="links"><a href="summary_seedlevel_bootstrap_ci.csv">summary_seedlevel_bootstrap_ci.csv</a></div></section>
<section><h2>八、5种子类别级分析与平均混淆矩阵</h2>{build_class_level()}<div class="links"><a href="class_level_rejection_5seed_summary.csv">class_level_rejection_5seed_summary.csv</a><a href="weak_or_high_rejection_classes_5seed.csv">weak_or_high_rejection_classes_5seed.csv</a><a href="class_level_5seed_matrices/">class_level_5seed_matrices/</a></div></section>
<section><h2>九、离散风险—覆盖率操作点</h2><p class="muted">当前结果为离散操作点，不是连续AURC曲线。</p>{build_risk_coverage()}<div class="links"><a href="risk_coverage_discrete_operating_points.csv">risk_coverage_discrete_operating_points.csv</a></div></section>
<section><h2>十、完整M与九类子集对照</h2>{build_m_compare(clean)}<div class="note warn">M_new_drop5_drop7在论文中建议表述为“预先整理的九类子集”，并与完整11类M_new同步报告。对于Potato - nematode可说明其样本量明显较少且属于线虫危害；对于Rice - leaf blast，若没有额外质量异常证据，不要声称其为错误或低质量类别。不能写成“因为准确率低所以删类”。</div><div class="links"><a href="protocolA_M_dataset_exclusion_note.md">protocolA_M_dataset_exclusion_note.md</a></div></section>
<section><h2>十一、运行时间、环境与方法参数</h2>{build_timing()}<div class="links"><a href="protocolA_method_parameters.md">protocolA_method_parameters.md</a><a href="environment_and_files.csv">environment_and_files.csv</a><a href="timing_vnew_seed11.csv">timing_vnew_seed11.csv</a></div></section>
<section><h2>十二、论文图件目录与文件对应表</h2>{build_figure_table()}<div class="links"><a href="README_figure_manifest.md">README_figure_manifest.md</a></div></section>
<section><h2>十三、固定讨论口径</h2><div class="note">F_new为天花板数据集，不能作为all3提升准确率的主要证据。V_new和M_new_drop5_drop7是多算法共识筛选高风险样本的核心支撑。高acc_kept但ARI不接近1，说明方法形成高纯度视觉子簇，而非恢复病害类别与簇的一一对应。UMAP主要改善复杂数据集中的覆盖率—风险平衡，而非在所有数据集和所有指标上全面领先。</div></section>
</main></body></html>"""


def main() -> None:
    html_text = build_html()
    (OUT / "index.html").write_text(html_text, encoding="utf-8")
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
