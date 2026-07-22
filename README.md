# Plant Disease Selective Unsupervised Clustering

This is the lightweight GitHub archive for the manuscript on transductive selective unsupervised clustering of plant disease images.

The repository contains the code, SHA256 audit lists, random-seed-level result tables, manuscript figures, HTML reports, and reproduction instructions. The cleaned image datasets and model checkpoints are intentionally **not** stored in Git because they are several GB in total. Put those assets back locally before rerunning experiments; see `MANUAL_ASSETS.md`.

## Protocol

The experiments use Protocol A: transductive selective unsupervised clustering. Disease labels are not used during feature extraction, dimensionality reduction, clustering, cross-algorithm cluster alignment, or rejection. Labels are used only after the clustering and rejection outputs are fixed, for many-to-one post-hoc alignment and external evaluation.

## Repository Layout

```text
code/                         Reproducibility scripts
code/legacy_original_deploy/  Earlier deploy scripts and dataset-building helpers
data/                         Manifests and dataset placement notes
data_audit/                   SHA256 exact-duplicate lists, pHash audit CSVs, label maps
docs/                         Method parameters, dataset notes, manuscript notes
external_baselines/dinov2/    DINOv2 seed-level results, summaries, reports, bootstrap CI
figures/                      Manuscript-ready figure files
manuscript/                   Manuscript DOCX copy
reports/                      Final HTML report and final summary tables
results/                      Seed-level raw and summary experimental CSVs
```

## Included

- Protocol A scripts for ConvNeXt feature extraction, main experiments, final analyses, reports, and figures.
- Earlier deploy scripts used during dataset construction and result generation.
- SHA256 duplicate-removal records and pHash near-duplicate risk-audit records.
- Five-seed main result CSVs, ablations, parameter sensitivity, fair dimensionality-reduction comparison, bootstrap confidence intervals, class-level statistics, and confusion matrices.
- DINOv2 external-baseline seed-level results and paired bootstrap CI.
- Final manuscript figures and HTML reports.

## Not Included In Git

- `data/images_clean/`: cleaned image datasets.
- `models/model.safetensors`: local ConvNeXt checkpoint.
- `models/dinov2_vit_base_patch14_lvd142m.safetensors`: DINOv2 checkpoint.

These files should be provided separately, for example by GitHub Releases, cloud drive, institutional repository, or local copy. Their required paths and checksums are listed in `MANUAL_ASSETS.md` and `WEIGHTS_AND_CHECKPOINTS.md`.

## Full Asset Package

The complete local reproducibility asset package, including the cleaned image datasets and model checkpoints, is available through Baidu Netdisk:

```text
Package: github_repro_archive_plant_disease_selective_clustering_20260721
URL: https://pan.baidu.com/s/1d8sLfgtz79Fjs-s7zvm2YQ?pwd=1234
Extraction code: 1234
```

After downloading, place the cleaned dataset folders under `data/images_clean/` and the checkpoint files under `models/` as described in `MANUAL_ASSETS.md`.

## Public Download Sources

The cleaned datasets in this project were reorganized from public plant-leaf image datasets. Download the original datasets from the sources below, then use the scripts and manifests in this repository to rebuild the flat folders expected by the experiments.

| Local folder | Source dataset | Public download page |
|---|---|---|
| `F_new` | PlantVillage fruit disease subset | https://github.com/spMohanty/PlantVillage-Dataset |
| `V_new` | PlantVillage vegetable disease subset | https://github.com/spMohanty/PlantVillage-Dataset |
| `M_new` | Multi-Crop Leaf Disease Dataset: Corn, Potato, Rice, Tomato, and Cashew | https://data.mendeley.com/datasets/z6jp232g5j |
| `M_new_drop5_drop7` | 9-class subset derived from `M_new` | https://data.mendeley.com/datasets/z6jp232g5j |
| `G_new` | Plant Leaf Disease Recognition Dataset, `Data for Leaf Disease/Background Removed` | https://www.kaggle.com/datasets/truongdinhit/plant-leaf-disease-recognition-dataset |

Additional metadata pages that may be useful:

- PlantVillage in TensorFlow Datasets: https://www.tensorflow.org/datasets/catalog/plant_village
- Plant Leaf Disease Recognition Dataset on Mendeley Data: https://data.mendeley.com/datasets/5g238dv4ht

Model/checkpoint download notes:

| File expected by scripts | Public source / status |
|---|---|
| `models/dinov2_vit_base_patch14_lvd142m.safetensors` | https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m |
| `models/model.safetensors` | Public timm ConvNeXt-XLarge ImageNet-22K checkpoint, corresponding to `convnext_xlarge.fb_in22k` / legacy `convnext_xlarge_in22k`: https://huggingface.co/timm/convnext_xlarge.fb_in22k. The original Facebook/timm download URL is `https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth`. The manuscript archive stores this checkpoint locally as `models/model.safetensors`; verify the archived local file with the SHA256 in `WEIGHTS_AND_CHECKPOINTS.md`. |

## Main Reports

```text
reports/protocolA_final_submission_v3/index.html
external_baselines/dinov2/reports/web_report_external_baselines_summary/index.html
external_baselines/dinov2/bootstrap_seedlevel/index.html
```

## Reproducibility

Use `REPRODUCE.md` for the command-level workflow and manuscript-result mapping.
