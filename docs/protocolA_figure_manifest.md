# Protocol A 论文图件清单

服务器完整图件目录：

`/data1/D/hostal/protocolA_manuscript_figures_20260720`

每张图均已导出为 SVG、PDF、TIFF 三种格式。

| 图号 | 文件前缀 | 用途 |
|---|---|---|
| Fig. 1 | `fig1_protocolA_workflow` | 方法流程图：图像、ConvNeXt、UMAP、三聚类、Hungarian对齐、共识拒识、事后评价 |
| Fig. 2 | `fig2_main_results_all3` | F、V、M_clean主实验：`acc_kept`、coverage、overall |
| Fig. 3 | `fig3_baseline_acc_kept` | KMeans、Birch、Agglomerative、any2、随机同覆盖率拒识、KMeans距离拒识、all3基线对比 |
| Fig. 4 | `fig4_discrete_risk_coverage` | 离散风险—覆盖率操作点 |
| Fig. 5 | `fig5_parameter_sensitivity_heatmap` | UMAP维度 × 聚类簇数参数敏感性热力图 |
| Fig. 6 | `fig6_class_level_weak_points` | 5种子类别级薄弱类别与拒识率 |
| Fig. 7 | `fig7_mclean_mean_confusion_matrices` | M_clean五种子平均保留样本混淆矩阵与带Rejected列的全样本矩阵 |
| Fig. 8 | `fig8_fair_reduction_comparison` | Raw 2048维、PCA 100维、UMAP 100维公平降维对照 |

建议正文主图：

- Fig. 1 方法流程图
- Fig. 2 主实验结果
- Fig. 3 或 Fig. 4 方法收益与风险覆盖率
- Fig. 7 M_clean混淆矩阵
- Fig. 8 公平降维对照可放正文或补充材料

建议补充材料：

- Fig. 5 参数敏感性
- Fig. 6 类别级薄弱类别
