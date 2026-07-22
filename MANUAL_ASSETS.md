# Manual Assets Required For Full Reruns

This lightweight GitHub repository does not include the cleaned image datasets or model checkpoints. To rerun the manuscript experiments from images, place the assets below after cloning the repository.

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

| Dataset folder | Role in manuscript | Expected image count |
|---|---|---:|
| `F_new` | PlantVillage fruit disease subset | 13950 |
| `V_new` | PlantVillage vegetable disease subset | 20214 |
| `M_new` | complete 11-class multi-crop disease dataset | 4842 |
| `M_new_drop5_drop7` | quality-control 9-class multi-crop disease subset | 4440 |
| `G_new` | auxiliary cross-crop visual category dataset | 3730 |

Use `data/images_clean_file_manifest_sha256.csv` to verify the expected file names and SHA256 values.

## Model Checkpoints

Place the checkpoints under `models/`.

| File | Purpose | SHA256 |
|---|---|---|
| `models/model.safetensors` | ConvNeXt-XLarge checkpoint used as frozen feature extractor | `72b257ce7a079089c1bac54151807caf1b10d33a570fa9738a3ba437d24fc4d9` |
| `models/dinov2_vit_base_patch14_lvd142m.safetensors` | DINOv2 external-baseline checkpoint | `55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b` |

## Suggested Manual Distribution

For GitHub, keep the repository lightweight and upload these large assets separately, for example:

- GitHub Releases, if file-size and quota limits are acceptable;
- an institutional data repository;
- Zenodo/OSF/Figshare;
- a private cloud drive during internal review.

After downloading or copying the assets, run the checksum verification commands in `REPRODUCE.md` before rerunning experiments.
