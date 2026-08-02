# Plant Disease Selective Unsupervised Clustering

This repository archives the reproducibility code, SHA256 de-duplication records,
seed-level results, confusion-matrix source files, and manuscript figure scripts
for the plant-disease selective unsupervised clustering manuscript.

The frozen final experiment version is **V13 / finally3**:

- Protocol: transductive selective unsupervised clustering.
- Backbone: frozen ConvNeXt-XLarge feature extractor.
- UMAP: `n_components=100`, `n_neighbors=15`, `min_dist=0.1`, `metric=euclidean`.
- Clustering: `K=60`.
- Seeds: `11, 22, 33, 44, 55`.
- Final datasets: `F_new`, `V_new`, complete `M_new` (MCLD-11), and `G_new`.

Disease labels are not used during feature extraction, dimensionality reduction,
clustering, cross-algorithm cluster alignment, or rejection. Labels are used only
after all clustering and rejection outputs are fixed, for many-to-one post-hoc
alignment and external clustering evaluation.

## Repository Layout

```text
code/                         V13/K60 reproduction scripts
data/                         Dataset manifests and expected clean-data layout
data_audit/                   SHA256 exact-duplicate and pHash risk-audit files
docs/                         V13/K60 method and figure notes
external_baselines/dinov2/    DINOv2 baseline summaries and reports
figures/v13_figure_report/    V13 figure source data and PNG/PDF figures
models/                       Expected local checkpoint paths
results/final_k60/            Final K=60 seed-level CSVs and matrix source files
outputs/                      Default location for newly generated outputs
```

## Large Assets

The cleaned images and model weights are large. They are provided separately in
the shared reproducibility package:

[Baidu Netdisk package](https://pan.baidu.com/s/1d8sLfgtz79Fjs-s7zvm2YQ?pwd=1234)

Extraction code: `1234`

After downloading, place or keep the assets at:

```text
data/images_clean/F_new/
data/images_clean/V_new/
data/images_clean/M_new/
data/images_clean/G_new/
models/model.safetensors
models/dinov2_vit_base_patch14_lvd142m.safetensors
```

`models/model.safetensors` corresponds to the public timm
`convnext_xlarge_in22k` ConvNeXt-XLarge ImageNet-22K checkpoint used as a frozen
2048-D feature extractor. DINOv2 uses the public
`vit_base_patch14_dinov2.lvd142m` checkpoint.

## Final Results

The final archived tables are in `results/final_k60/`:

```text
main_results.csv
ablation_results.csv
all_vs_retained_metrics.csv
bootstrap_results.csv
classwise_results.csv
runtime_results.csv
confusion_matrices/
```

The final manuscript figures for quick inspection are in:

```text
figures/v13_figure_report/reproduced_figures/
```

For exact rerun commands and manuscript mapping, see `REPRODUCE.md`.
