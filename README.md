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

## Main Reports

```text
reports/protocolA_final_submission_v3/index.html
external_baselines/dinov2/reports/web_report_external_baselines_summary/index.html
external_baselines/dinov2/bootstrap_seedlevel/index.html
```

## Reproducibility

Use `REPRODUCE.md` for the command-level workflow and manuscript-result mapping.
