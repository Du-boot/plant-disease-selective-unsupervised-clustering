# Plant Disease Selective Unsupervised Clustering

This repository archives the code, audit lists, seed-level results, reports, and figure scripts used for the manuscript:

`植物病害选择性无监督聚类_中文论文完善稿_V10_数据许可与权重溯源确认版.docx`

The experiments follow Protocol A: transductive selective unsupervised clustering. Disease labels are not used during feature extraction, dimensionality reduction, clustering, cross-algorithm cluster alignment, or rejection. Labels are used only after the clustering/rejection outputs are fixed, for many-to-one post-hoc label alignment and external evaluation.

## Repository Layout

```text
code/                         Reproducibility scripts
code/legacy_original_deploy/  Original deploy scripts and dataset-building helpers
data_audit/                   SHA256 exact-duplicate lists, pHash audit CSVs, label maps
data/images_clean/            SHA256-cleaned image datasets used by the manuscript
docs/                         Method parameters, dataset notes, manuscript draft notes
external_baselines/dinov2/    DINOv2 external-baseline seed results and reports
figures/                      Manuscript figure outputs and figure scripts
manuscript/                   Current manuscript DOCX copy
models/                       ConvNeXt and DINOv2 checkpoints required for reproduction
outputs/                      Placeholder for newly generated outputs; not tracked
reports/                      Final HTML report and final summary tables
results/                      Seed-level raw and summary experimental CSVs
data/                         Placeholder for datasets; not tracked
```

## What Is Included

- All Protocol A Python scripts used for duplicate auditing, main experiments, final analyses, reports, and figures.
- Original deploy scripts used earlier in the project, including `encode.py`, `get_final_result.py`, and `make_confusion_m.py`.
- SHA256 exact-duplicate removal lists and pHash near-duplicate risk-audit files.
- Random-seed-level results for the main ConvNeXt pipeline.
- DINOv2 external-baseline seed-level results, summaries, and seed-level paired bootstrap confidence intervals.
- Final manuscript figures and HTML reports.

## What Is Not Included

The local archive now includes the SHA256-cleaned image datasets under `data/images_clean/`, the ConvNeXt checkpoint under `models/model.safetensors`, and the DINOv2 checkpoint under `models/dinov2_vit_base_patch14_lvd142m.safetensors`, so the manuscript experiments can be rerun from images.

Because the clean images and checkpoint are several GB in total, this repository includes `.gitattributes` rules for Git LFS. Before uploading to GitHub, run `git lfs install`.

## Main Results To Cite

The frozen final report is:

```text
reports/protocolA_final_submission_v3/index.html
```

The DINOv2 external-baseline report is:

```text
external_baselines/dinov2/reports/web_report_external_baselines_summary/index.html
```

The DINOv2 seed-level paired bootstrap report is:

```text
external_baselines/dinov2/bootstrap_seedlevel/index.html
```

## Reproducibility

For commands and expected paths, see `REPRODUCE.md`.
