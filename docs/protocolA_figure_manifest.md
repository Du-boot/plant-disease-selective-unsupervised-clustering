# V13/K60 Paper Figure Manifest

Final archived figures are stored in:

```text
figures/v13_figure_report/reproduced_figures/
```

Their source data are stored in:

```text
figures/v13_figure_report/source_data/
```

All numerical results use five seeds (`11, 22, 33, 44, 55`) under the frozen
V13 configuration: UMAP 100-D, `n_neighbors=15`, `min_dist=0.1`, and `K=60`.
Error bars show the seed-level standard deviation. Confusion matrices are
row-normalized means across the five seeds.

## Main Figures

| Figure | File | Content | Source data |
|---|---|---|---|
| Fig. 1 | `Main_Figures/Fig01_Detailed_Framework.png/.pdf` | Transductive selective unsupervised clustering workflow | Script-defined workflow |
| Fig. 2 | `Main_Figures/Fig02_Representative_Dataset_Images.png/.pdf` | Representative original images from the four datasets | `source_data/representative_images/` |
| Fig. 3 | `Main_Figures/Fig03_PostHoc_ManyToOne_Mapping.png/.pdf` | Cluster alignment and post-hoc many-to-one mapping | Script-defined schematic |
| Fig. 4 | `Main_Figures/Fig04_Main_C3_Results.png/.pdf` | C3/all3 main results across four datasets | `source_data/main_results/clean_main.csv` |
| Fig. 5 | `Main_Figures/Fig05_All_vs_Retained_Clustering_Metrics.png/.pdf` | ARI/NMI/AMI for all versus retained samples | `source_data/main_results/summary_all_vs_kept_cluster_metrics.csv` |
| Fig. 6 | `Main_Figures/Fig06_Fair_Dimensionality_Reduction.png/.pdf` | Fair Raw/PCA/UMAP comparison | `source_data/main_results/fair_reduction_5seeds.csv` |
| Fig. 7 | `Main_Figures/Fig07_Parameter_Sensitivity.png/.pdf` | UMAP and cluster-number sensitivity | `source_data/main_results/parameter_sensitivity_5seeds.csv` |
| Fig. 8 | `Main_Figures/Fig08_External_DINOv2_Baseline.png/.pdf` | DINOv2 external self-supervised baseline | `source_data/dinov2/summary_convnext_vs_dinov2_key.csv` |
| Fig. 9 | `Main_Figures/Fig09_Weak_and_HighRejection_Classes.png/.pdf` | Class-level weak and high-rejection categories | `source_data/confusion_matrices/class_level_rejection_5seed_summary.csv` |
| Fig. 10a | `Main_Figures/Fig10a_PV_Fruit_Kept_Confusion_Matrix.png/.pdf` | PV-Fruit retained-sample confusion matrix | `source_data/confusion_matrices/PV-Fruit_kept.csv` |
| Fig. 10b | `Main_Figures/Fig10b_PV_Vegetable_Kept_Confusion_Matrix.png/.pdf` | PV-Vegetable retained-sample confusion matrix | `source_data/confusion_matrices/PV-Vegetable_kept.csv` |
| Fig. 10c | `Main_Figures/Fig10c_MCLD_11_Kept_Confusion_Matrix.png/.pdf` | MCLD-11 retained-sample confusion matrix | `source_data/confusion_matrices/MCLD-11_kept.csv` |
| Fig. 10d | `Main_Figures/Fig10d_DFLD_BR_4_Kept_Confusion_Matrix.png/.pdf` | DFLD-BR-4 retained-sample confusion matrix | `source_data/confusion_matrices/DFLD-BR-4_kept.csv` |
| Fig. 11 | `Main_Figures/Fig11_Discrete_Risk_Coverage_Points.png/.pdf` | Discrete risk-coverage operating points, not a continuous AURC curve | `source_data/main_results/risk_coverage_discrete_operating_points.csv` |

## Supplementary Figures

| Figure | File | Content |
|---|---|---|
| Fig. S01a | `Supplementary_Figures/FigS01a_PV_Fruit_AllPlusRejected_Confusion_Matrix.png/.pdf` | PV-Fruit all-sample matrix with a Rejected column |
| Fig. S01b | `Supplementary_Figures/FigS01b_PV_Vegetable_AllPlusRejected_Confusion_Matrix.png/.pdf` | PV-Vegetable all-sample matrix with a Rejected column |
| Fig. S01c | `Supplementary_Figures/FigS01c_MCLD_11_AllPlusRejected_Confusion_Matrix.png/.pdf` | MCLD-11 all-sample matrix with a Rejected column |
| Fig. S01d | `Supplementary_Figures/FigS01d_DFLD_BR_4_AllPlusRejected_Confusion_Matrix.png/.pdf` | DFLD-BR-4 all-sample matrix with a Rejected column |

The confusion matrices are separate files rather than a four-panel composite so
that all real class labels and non-zero cell annotations remain readable.
