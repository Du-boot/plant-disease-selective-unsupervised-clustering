# Reproduction Guide

This guide reproduces the frozen V13/finally3 manuscript experiments.

Final configuration:

```text
Datasets: F_new, V_new, M_new, G_new
UMAP: n_components=100, n_neighbors=15, min_dist=0.1
Clusters: K=60
Seeds: 11, 22, 33, 44, 55
Protocol: transductive selective unsupervised clustering
```

`M_new` is the complete 11-class MCLD dataset after SHA256 exact-duplicate
removal.

## 0. Prepare Assets

Download the large asset package:

[Baidu Netdisk package](https://pan.baidu.com/s/1d8sLfgtz79Fjs-s7zvm2YQ?pwd=1234)

Extraction code: `1234`

Expected local paths after unpacking:

```text
data/images_clean/F_new/
data/images_clean/V_new/
data/images_clean/M_new/
data/images_clean/G_new/
models/model.safetensors
models/dinov2_vit_base_patch14_lvd142m.safetensors
```

The SHA256 image manifest is:

```text
data/images_clean_file_manifest_sha256.csv
```

SHA256 exact duplicates were removed. pHash files are included only as
near-duplicate risk-audit records and are not used for deletion, clustering,
rejection, or evaluation.

## 1. Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

The original full rerun used an RTX 4090 GPU. Feature extraction benefits most
from GPU acceleration; the clustering stages are also CPU/RAM intensive.

## 2. Extract ConvNeXt Features

From the repository root:

```bash
python code/extract_convnext_features.py
```

Default outputs:

```text
outputs/convnext_features/features/F_new.npy
outputs/convnext_features/features/V_new.npy
outputs/convnext_features/features/M_new.npy
outputs/convnext_features/features/G_new.npy
```

Manuscript mapping: frozen ConvNeXt-XLarge 2048-D visual feature extraction.

## 3. Run Protocol A Main Experiments

```bash
python code/protocolA_followup_experiments.py --stage all
```

Default outputs:

```text
outputs/protocolA_followup/duplicate_audit/
outputs/protocolA_followup/followup_results/clean_main.csv
outputs/protocolA_followup/followup_results/random_same_coverage_completed.csv
outputs/protocolA_followup/followup_results/fair_reduction_5seeds.csv
outputs/protocolA_followup/followup_results/parameter_sensitivity_5seeds.csv
outputs/protocolA_followup/followup_results/internal_metrics.csv
```

Manuscript mapping:

| Manuscript content | Output file |
|---|---|
| Main all3 results | `clean_main.csv` |
| KMeans/Birch/Agglomerative/any2/all3 ablation | `clean_main.csv` |
| KMeans distance rejection | `clean_main.csv` |
| Same-coverage random rejection | `random_same_coverage_completed.csv` |
| Fair Raw/PCA/UMAP comparison | `fair_reduction_5seeds.csv` |
| Parameter sensitivity | `parameter_sensitivity_5seeds.csv` |
| Internal clustering metrics | `internal_metrics.csv` |

Expected V13 all3 means from `results/final_k60/main_results.csv`:

| Dataset | Manuscript name | Acc_kept | Coverage |
|---|---|---:|---:|
| `F_new` | PV-Fruit | 99.95% +/- 0.02% | 61.37% +/- 2.06% |
| `V_new` | PV-Vegetable | 99.07% +/- 0.24% | 82.62% +/- 5.67% |
| `M_new` | MCLD-11 | 90.28% +/- 0.64% | 73.36% +/- 1.51% |
| `G_new` | DFLD-BR-4 | 99.87% +/- 0.02% | 64.41% +/- 1.50% |

## 4. Final Statistical Analyses And Matrices

```bash
python code/protocolA_final_required_patch.py
```

Default outputs:

```text
outputs/protocolA_final_required_patch/class_level_5seed/
outputs/protocolA_final_required_patch/bootstrap_seedlevel/
outputs/protocolA_final_required_patch/risk_coverage_discrete/
```

Manuscript mapping:

| Manuscript content | Output directory |
|---|---|
| Five-seed class-level accuracy/rejection | `class_level_5seed/` |
| Retained-sample average confusion matrices | `class_level_5seed/` |
| All-sample matrices with Rejected column | `class_level_5seed/` |
| Seed-level paired Bootstrap 95% CI | `bootstrap_seedlevel/` |
| Discrete risk-coverage operating points | `risk_coverage_discrete/` |

Each per-sample prediction CSV contains:

```text
dataset, seed, true_label, aligned_pred_label, kept
```

## 5. Runtime And Supplementary Materials

```bash
python code/protocolA_final_supplements.py --stage all
```

Manuscript mapping: method-parameter notes, runtime/environment summaries,
pHash risk-audit assets, and supplementary consistency checks.

## 6. DINOv2 External Baselines

```bash
python code/external_dinov2_baselines.py
```

Manuscript mapping: public self-supervised DINOv2 feature baseline and DINOv2
backbone transfer test under the same `K=60` setting.

Expected DINOv2 all3 means:

| Dataset | Method | Acc_kept | Coverage |
|---|---|---:|---:|
| `F_new` | DINOv2 + UMAP + all3 | 99.84% | 56.42% |
| `V_new` | DINOv2 + UMAP + all3 | 98.51% | 77.24% |
| `M_new` | DINOv2 + UMAP + all3 | 86.96% | 68.01% |
| `G_new` | DINOv2 + UMAP + all3 | 99.91% | 60.52% |

## 7. Rebuild Figures

```bash
python code/rebuild_v13_figures.py
python code/draw_individual_confusion_matrices_server.py
```

Default figure outputs:

```text
figures/v13_figure_report/reproduced_figures/
outputs/individual_confusion_matrices_server_style/retained_samples/
outputs/individual_confusion_matrices_server_style/all_samples_with_rejected/
```

Archived V13 figure source data and PNG/PDF outputs are already stored at:

```text
figures/v13_figure_report/source_data/
figures/v13_figure_report/reproduced_figures/
```

High-resolution TIFF submission figures are intentionally not duplicated in the
GitHub-ready light archive because they are large; they are available in the
separate Baidu Netdisk package.

The final confusion matrices are intentionally exported as separate figures
(`Fig10a`--`Fig10d` and `FigS01a`--`FigS01d`). This preserves complete class
names and cell annotations that would become unreadable in a four-panel layout.

## 8. Verify Archived Final Tables Without Rerunning

```bash
python - <<'PY'
from pathlib import Path
import csv, statistics

rows = list(csv.DictReader(open(Path("results/final_k60/main_results.csv"), encoding="utf-8")))
for ds in ["F_new", "V_new", "M_new", "G_new"]:
    sub = [r for r in rows if r["dataset_code"] == ds and r["method"] == "all3" and r["n_clusters"] == "60"]
    acc = [float(r["acc_kept"]) for r in sub]
    cov = [float(r["coverage"]) for r in sub]
    print(ds, len(sub), statistics.mean(acc) * 100, statistics.stdev(acc) * 100,
          statistics.mean(cov) * 100, statistics.stdev(cov) * 100)
PY
```

This should reproduce the expected V13 all3 means listed above.
