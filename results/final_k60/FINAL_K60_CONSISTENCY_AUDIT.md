# finally3 K=60 结果一致性审计

审计时间：2026-07-26

最终配置：

- UMAP `n_components=100`
- UMAP `n_neighbors=15`
- UMAP `min_dist=0.1`
- 聚类簇数 `K=60`
- 随机种子：11、22、33、44、55
- 数据集：F_new、V_new、M_new、G_new

## 已修正的问题

1. `protocolA_final_supplements.fit_or_load_main()` 原默认 `k=20`，导致类别级混淆矩阵曾读取旧 K=20 缓存。已改为 `k=60`，并重跑：
   - `protocolA_final_required_patch.py`
   - `protocolA_final_supplements.py`
   - `rebuild_v12_figures.py`

2. `all4_unified_grid_diagnostic.py` 注释写四个数据集，但实际 `DATASETS` 曾漏掉 `M_new`。已改为：
   - `["F_new", "V_new", "M_new", "G_new"]`
   并重跑 all4 诊断。

## 核查结果

- `clean_main.csv`：120 行，4 数据集 x 5 seeds x 6 methods，唯一 `K=60`。
- `random_same_coverage_completed.csv`：400 行，4 数据集 x 5 seeds x 20 repeats，唯一 `K=60`。
- `fair_reduction_5seeds.csv`：60 行，4 数据集 x 5 seeds x raw/PCA/UMAP，唯一 `K=60`。
- `external_dinov2_results.csv`：120 行，4 数据集 x 5 seeds x 6 methods，唯一 `K=60`。
- `all4_grid_seedlevel.csv`：60 行，4 数据集 x 5 seeds x 3 methods，`n_neighbors=15`，`min_dist=0.1`，唯一 `K=60`。
- `v12_figure_report/source_data/main_results/clean_main.csv` 与 `protocolA_followup/followup_results/clean_main.csv` 完全一致。
- Fig10 的 MCLD-11 source matrix 与 `protocolA_final_required_patch` 的 5-seed 平均矩阵一致。

## K=60 C3/all3 主结果

| dataset | Acc_kept mean | Acc_kept SD | Coverage mean | Coverage SD | Conservative mean |
|---|---:|---:|---:|---:|---:|
| F_new | 0.999487 | 0.000151 | 0.613749 | 0.020563 | 0.613434 |
| V_new | 0.990697 | 0.002442 | 0.826190 | 0.056677 | 0.818423 |
| M_new | 0.902801 | 0.006442 | 0.733622 | 0.015133 | 0.662288 |
| G_new | 0.998666 | 0.000202 | 0.644075 | 0.015013 | 0.643217 |

## MCLD-11 关键矩阵核查

5-seed retained matrix 中：

- `Potato - nematode -> Potato - nematode = 0.774487`
- `Potato - nematode -> Potato - fungi = 0.225513`

seed=11 supplement matrix 中：

- `Potato - nematode` retained accuracy = 0.758065
- rejection rate = 0.088235

结论：finally3 当前主结果、外部基线、混淆矩阵、图件 source data 和 all4 诊断已统一到 K=60 口径。
