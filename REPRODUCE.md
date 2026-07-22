# Reproduction Guide

This archive is designed to reproduce the manuscript experiments from the included clean images, code, and ConvNeXt checkpoint.

The manuscript protocol is **Protocol A: transductive selective unsupervised clustering**. Labels are not used during feature extraction, dimensionality reduction, clustering, cross-algorithm cluster alignment, or rejection. Labels are used only after results are fixed for many-to-one post-hoc evaluation.

## 0. Prepare Git LFS Before Uploading

The archive contains image datasets and `models/model.safetensors`. Use Git LFS before uploading to GitHub:

```bash
git lfs install
git add .gitattributes
git add data/images_clean models/model.safetensors
```

Without Git LFS, GitHub will reject `model.safetensors` because it is larger than 100 MB.

## 1. Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

GPU is strongly recommended for feature extraction. The original run used RTX 4090 GPUs.

## 2. Included Data And Weight

Clean images are included at:

```text
data/images_clean/F_new/
data/images_clean/V_new/
data/images_clean/G_new/
data/images_clean/M_new/
data/images_clean/M_new_drop5_drop7/
```

The ConvNeXt checkpoint is included at:

```text
models/model.safetensors
```

The image manifest is:

```text
data/images_clean_file_manifest_sha256.csv
```

## 3. Full Reproduction From Images

Run all commands from the repository root.

### Step 1: Extract ConvNeXt Features

This command reproduces the frozen ConvNeXt feature-extraction stage used by the main manuscript experiments:

```bash
python code/extract_convnext_features.py \
  --data-root data/images_clean \
  --checkpoint models/model.safetensors \
  --feature-root outputs/convnext_feature_root \
  --datasets F_new V_new M_new_drop5_drop7 G_new M_new
```

Output:

```text
outputs/convnext_feature_root/features/F_new.npy
outputs/convnext_feature_root/features/V_new.npy
outputs/convnext_feature_root/features/M_new_drop5_drop7.npy
outputs/convnext_feature_root/features/G_new.npy
outputs/convnext_feature_root/features/M_new.npy
```

This corresponds to the manuscript method section:

```text
Frozen ConvNeXt visual feature extraction -> 2048-D features
```

### Step 2: Run Protocol A Main Experiments

This command reproduces the main Protocol A experiments for `F_new`, `V_new`, `M_new_drop5_drop7`, and `G_new`:

```bash
python code/protocolA_followup_experiments.py \
  --data-root data/images_clean \
  --feature-root outputs/convnext_feature_root \
  --out outputs/protocolA_followup_20260720 \
  --stage all
```

Output:

```text
outputs/protocolA_followup_20260720/duplicate_audit/
outputs/protocolA_followup_20260720/followup_results/clean_main.csv
outputs/protocolA_followup_20260720/followup_results/fair_reduction_5seeds.csv
outputs/protocolA_followup_20260720/followup_results/parameter_sensitivity_5seeds.csv
outputs/protocolA_followup_20260720/followup_results/random_same_coverage_completed.csv
```

This corresponds to manuscript sections/tables:

| Manuscript content | Output file |
|---|---|
| Main 5-seed result: F_new, V_new, MCLD-9 | `clean_main.csv` |
| KMeans/Birch/Agglomerative/any2/all3/distance/random ablation | `clean_main.csv`, `random_same_coverage_completed.csv` |
| Parameter sensitivity | `parameter_sensitivity_5seeds.csv` |
| Fair Raw/PCA/UMAP reduction comparison | `fair_reduction_5seeds.csv` |
| SHA256 and pHash audit | `duplicate_audit/` |

Expected article-level main numbers from the archived final run:

| Dataset | Manuscript name | Acc_kept | Coverage |
|---|---|---:|---:|
| `F_new` | PV-Fruit | 99.95% ± 0.00% | 78.70% ± 4.58% |
| `V_new` | PV-Vegetable | 97.85% ± 2.10% | 89.99% ± 4.34% |
| `M_new_drop5_drop7` | MCLD-9 | 90.72% ± 1.48% | 90.61% ± 5.05% |

### Step 3: Run Class-Level, Bootstrap, Risk-Coverage, And MCLD-11 Analyses

```bash
python code/protocolA_final_required_patch.py \
  --data-root data/images_clean \
  --feature-root outputs/convnext_feature_root \
  --followup-root outputs/protocolA_followup_20260720 \
  --out outputs/protocolA_final_required_patch_20260720
```

This corresponds to manuscript sections/tables:

| Manuscript content | Output directory/file |
|---|---|
| 5-seed class-level rejection statistics | `class_level_5seed/` |
| 5-seed average confusion matrices | `class_level_5seed/` |
| Seed-level paired Bootstrap 95% CI | `bootstrap_seedlevel/` |
| Discrete risk-coverage operating points | `risk_coverage_discrete/` |
| MCLD-11 full-dataset robustness result | `m_new_full/M_new_full_main.csv` |

Expected MCLD-11 article number:

| Dataset | Manuscript name | Acc_kept | Coverage |
|---|---|---:|---:|
| `M_new` | MCLD-11 | 87.38% ± 0.76% | 81.09% ± 0.89% |

### Step 4: Rebuild Manuscript Figures

This script can rebuild the main figure set from the archived CSVs:

```bash
python code/build_protocolA_figures.py
```

Output:

```text
outputs/manuscript_figures/
```

Archived generated figures are already stored in:

```text
figures/protocolA_manuscript_figures_20260720/
```

Figure mapping:

| Figure | Content | Script |
|---|---|---|
| Fig. 1 | Protocol A workflow | `fig1_workflow()` |
| Fig. 2 | Main all3 results | `fig2_main_results()` |
| Fig. 3 | Ablation comparison | `fig3_baseline_comparison()` |
| Fig. 4 | Discrete risk-coverage operating points | `fig4_risk_coverage()` |
| Fig. 5 | Parameter sensitivity heatmap | `fig5_parameter_heatmap()` |
| Fig. 6 | Weak/high-rejection classes | `fig6_weak_classes()` |
| Fig. 7 | Average confusion matrix | `fig7_confusion()` |
| Fig. 8 | Fair reduction comparison | `fig8_fair_reduction()` |

## 4. DINOv2 External Baselines

DINOv2 weights are included locally at:

```text
models/dinov2_vit_base_patch14_lvd142m.safetensors
```

The script loads this local checkpoint by default. If the checkpoint is removed, `timm` can download it online; if direct HuggingFace access is unstable, use `HF_ENDPOINT=https://hf-mirror.com`.

Run:

```bash
python code/external_dinov2_baselines.py \
  --data-root data/images_clean \
  --out outputs/external_baselines_dinov2 \
  --audit-json data_audit/duplicate_audit/exact_clean_keep_names.json \
  --dinov2-checkpoint models/dinov2_vit_base_patch14_lvd142m.safetensors
```

This corresponds to manuscript external-baseline results:

| Dataset | Method | Acc_kept |
|---|---|---:|
| `V_new` | DINOv2 + UMAP + all3 | 95.60% ± 0.55% |
| `M_new_drop5_drop7` | DINOv2 + UMAP + all3 | 88.15% ± 0.73% |

Archived DINOv2 outputs:

```text
external_baselines/dinov2/external_dinov2_results.csv
external_baselines/dinov2/external_dinov2_summary.csv
external_baselines/dinov2/reports/web_report_external_baselines_summary/index.html
```

The seed-level paired Bootstrap CI table for DINOv2 is archived at:

```text
external_baselines/dinov2/bootstrap_seedlevel/dinov2_seedlevel_paired_bootstrap_ci.csv
```

It supports the manuscript statement that, under a public self-supervised DINOv2 backbone, all3 improves retained-sample quality over KMeans/any2 on the more complex V and M datasets, while reducing coverage.

## 5. Verify Archived Results Without Rerunning

To verify that the archived CSVs match the manuscript:

```bash
python - <<'PY'
from pathlib import Path
import csv, statistics, math
repo = Path('.')
def f(x):
    try: return float(x)
    except Exception: return math.nan

rows = list(csv.DictReader(open(repo/'results/followup_results_seedlevel/clean_main.csv', encoding='utf-8')))
for ds in ['F_new', 'V_new', 'M_new_drop5_drop7']:
    sub = [r for r in rows if r['dataset'] == ds and r['method'] == 'all3' and r['reduction'] == 'umap' and r['dim'] == '100' and r['k'] == '20']
    acc = [1 - f(r['kept_error']) if not r.get('acc_kept') else f(r['acc_kept']) for r in sub]
    cov = [1 - f(r['rejection_rate']) for r in sub]
    print(ds, len(sub), statistics.mean(acc)*100, statistics.stdev(acc)*100, statistics.mean(cov)*100, statistics.stdev(cov)*100)

rows = list(csv.DictReader(open(repo/'results/m_new_full/M_new_full_main.csv', encoding='utf-8')))
sub = [r for r in rows if r['dataset'] == 'M_new' and r['method'] == 'all3']
print('M_new', statistics.mean([f(r['acc_kept']) for r in sub])*100, statistics.stdev([f(r['acc_kept']) for r in sub])*100)
PY
```
