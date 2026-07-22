# Reproduction Guide

This lightweight GitHub repository contains the code, archived result tables, reports, and figure files. It does **not** include the cleaned image datasets or model checkpoints. To rerun the manuscript experiments from images, first place the manual assets described in `MANUAL_ASSETS.md`.

The manuscript protocol is **Protocol A: transductive selective unsupervised clustering**. Labels are not used during feature extraction, dimensionality reduction, clustering, cross-algorithm cluster alignment, or rejection. Labels are used only after results are fixed for many-to-one post-hoc evaluation.

## 1. Environment

```bash
pip install -r requirements.txt
```

GPU is recommended for feature extraction. The original timed run used an RTX 4090 GPU.

## 2. Restore Manual Assets

After cloning this repository, manually place the datasets and checkpoints:

```text
data/images_clean/F_new/
data/images_clean/V_new/
data/images_clean/G_new/
data/images_clean/M_new/
data/images_clean/M_new_drop5_drop7/

models/model.safetensors
models/dinov2_vit_base_patch14_lvd142m.safetensors
```

Expected counts and checksums are in:

```text
MANUAL_ASSETS.md
data/images_clean_file_manifest_sha256.csv
WEIGHTS_AND_CHECKPOINTS.md
```

Optional PowerShell checksum checks:

```powershell
Get-FileHash models/model.safetensors -Algorithm SHA256
Get-FileHash models/dinov2_vit_base_patch14_lvd142m.safetensors -Algorithm SHA256
```

Expected checkpoint SHA256 values:

```text
ConvNeXt: 72b257ce7a079089c1bac54151807caf1b10d33a570fa9738a3ba437d24fc4d9
DINOv2:   55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b
```

## 3. Rebuild ConvNeXt Features

Run from the repository root:

```bash
python code/extract_convnext_features.py \
  --data-root data/images_clean \
  --checkpoint models/model.safetensors \
  --feature-root outputs/convnext_feature_root \
  --datasets F_new V_new M_new_drop5_drop7 G_new M_new
```

This corresponds to the manuscript method step:

```text
image -> frozen ConvNeXt feature extractor -> 2048-D features
```

## 4. Run Protocol A Main Experiments

```bash
python code/protocolA_followup_experiments.py \
  --data-root data/images_clean \
  --feature-root outputs/convnext_feature_root \
  --out outputs/protocolA_followup_20260720 \
  --stage all
```

Main manuscript mapping:

| Manuscript content | Output file |
|---|---|
| 5-seed main results for F, V, MCLD-9, G | `outputs/protocolA_followup_20260720/followup_results/clean_main.csv` |
| KMeans/Birch/Agglomerative/any2/all3/distance/random ablation | `clean_main.csv`, `random_same_coverage_completed.csv` |
| Parameter sensitivity | `parameter_sensitivity_5seeds.csv` |
| Fair Raw/PCA/UMAP comparison | `fair_reduction_5seeds.csv` |
| SHA256 and pHash audit | `duplicate_audit/` |

Expected article-level main numbers from the frozen archived run:

| Dataset | Manuscript name | Acc_kept | Coverage |
|---|---|---:|---:|
| `F_new` | PV-Fruit | 99.95% +/- 0.00% | 78.70% +/- 4.58% |
| `V_new` | PV-Vegetable | 97.85% +/- 2.10% | 89.99% +/- 4.34% |
| `M_new_drop5_drop7` | MCLD-9 | 90.72% +/- 1.48% | 90.61% +/- 5.05% |

## 5. Run Final Analyses

```bash
python code/protocolA_final_required_patch.py \
  --data-root data/images_clean \
  --feature-root outputs/convnext_feature_root \
  --followup-root outputs/protocolA_followup_20260720 \
  --out outputs/protocolA_final_required_patch_20260720
```

Manuscript mapping:

| Manuscript content | Output directory/file |
|---|---|
| 5-seed class-level rejection statistics | `class_level_5seed/` |
| 5-seed average confusion matrices | `class_level_5seed/` |
| Seed-level paired bootstrap 95% CI | `bootstrap_seedlevel/` |
| Discrete risk-coverage operating points | `risk_coverage_discrete/` |
| Complete MCLD-11 robustness result | `m_new_full/M_new_full_main.csv` |

Expected MCLD-11 robustness number:

| Dataset | Manuscript name | Acc_kept | Coverage |
|---|---|---:|---:|
| `M_new` | MCLD-11 | 87.38% +/- 0.76% | 81.09% +/- 0.89% |

## 6. Rebuild Manuscript Figures

The main figure set can be rebuilt from archived CSVs:

```bash
python code/build_protocolA_figures.py
```

The already archived figure files are in:

```text
figures/protocolA_manuscript_figures_20260720/
```

## 7. Run DINOv2 External Baselines

After placing the DINOv2 checkpoint:

```bash
python code/external_dinov2_baselines.py \
  --data-root data/images_clean \
  --checkpoint models/dinov2_vit_base_patch14_lvd142m.safetensors \
  --out outputs/external_baselines_dinov2_20260721
```

Archived DINOv2 reports:

```text
external_baselines/dinov2/reports/web_report_external_baselines_summary/index.html
external_baselines/dinov2/bootstrap_seedlevel/index.html
```

Key DINOv2 external-baseline numbers:

| Dataset | DINOv2 UMAP all3 Acc_kept |
|---|---:|
| `V_new` | 95.60% +/- 0.55% |
| `M_new_drop5_drop7` | 88.15% +/- 0.73% |

## 8. Read Results Without Rerunning

The frozen manuscript results are already archived under:

```text
results/
reports/protocolA_final_submission_v3/index.html
external_baselines/dinov2/
```

Use these files when only checking manuscript tables and figures. Full reruns require the manual datasets and checkpoints.
