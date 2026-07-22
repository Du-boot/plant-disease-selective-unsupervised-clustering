# Manual Assets Required For Full Reruns

This lightweight GitHub repository does not include the cleaned image datasets or model checkpoints directly in Git. These large assets are provided through the Baidu Netdisk package below. To rerun the manuscript experiments from images, download the package and place the assets in the required local paths after cloning the repository.

## Full Asset Package

The complete local reproducibility asset package is available through Baidu Netdisk:

```text
Package: github_repro_archive_plant_disease_selective_clustering_20260721
URL: https://pan.baidu.com/s/1d8sLfgtz79Fjs-s7zvm2YQ?pwd=1234
Extraction code: 1234
```

This package should contain the cleaned image datasets and model checkpoints needed for full reruns. After downloading, place or copy the folders/files into the layout below.

## Required Directory Layout

```text
data/images_clean/
  F_new/
  V_new/
  M_new/
  M_new_drop5_drop7/
  G_new/

models/
  model.safetensors
  dinov2_vit_base_patch14_lvd142m.safetensors
```

## Clean Image Datasets

The final SHA256-cleaned datasets used by the manuscript should be placed under `data/images_clean/`.

| Dataset folder | Role in manuscript | Expected image count | Original public source |
|---|---|---:|---|
| `F_new` | PlantVillage fruit disease subset | 13950 | https://github.com/spMohanty/PlantVillage-Dataset |
| `V_new` | PlantVillage vegetable disease subset | 20214 | https://github.com/spMohanty/PlantVillage-Dataset |
| `M_new` | complete 11-class multi-crop disease dataset | 4842 | https://data.mendeley.com/datasets/z6jp232g5j |
| `M_new_drop5_drop7` | quality-control 9-class multi-crop disease subset | 4440 | https://data.mendeley.com/datasets/z6jp232g5j |
| `G_new` | auxiliary cross-crop visual category dataset | 3730 | https://www.kaggle.com/datasets/truongdinhit/plant-leaf-disease-recognition-dataset |

Use `data/images_clean_file_manifest_sha256.csv` to verify the expected file names and SHA256 values.

The `G_new` folder was organized from the `Data for Leaf Disease/Background Removed` part of the Plant Leaf Disease Recognition Dataset. A related Mendeley metadata page is available at https://data.mendeley.com/datasets/5g238dv4ht.

## Model Checkpoints

Place the checkpoints under `models/`.

| File | Purpose | SHA256 |
|---|---|---|
| `models/model.safetensors` | ConvNeXt-XLarge checkpoint used as frozen feature extractor | `72b257ce7a079089c1bac54151807caf1b10d33a570fa9738a3ba437d24fc4d9` |
| `models/dinov2_vit_base_patch14_lvd142m.safetensors` | DINOv2 external-baseline checkpoint | `55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b` |

DINOv2 can be downloaded from https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m.

ConvNeXt can be downloaded from the public timm ConvNeXt-XLarge ImageNet-22K checkpoint page: https://huggingface.co/timm/convnext_xlarge.fb_in22k. The original Facebook/timm URL is https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth. The manuscript archive stores the same public ConvNeXt-XLarge ImageNet-22K checkpoint locally as `models/model.safetensors`; verify the local archive file with the SHA256 hash above.

## Suggested Manual Distribution

For GitHub, keep the repository lightweight and upload these large assets separately, for example:

- GitHub Releases, if file-size and quota limits are acceptable;
- an institutional data repository;
- Zenodo/OSF/Figshare;
- a private cloud drive during internal review.

After downloading or copying the assets, run the checksum verification commands in `REPRODUCE.md` before rerunning experiments.
