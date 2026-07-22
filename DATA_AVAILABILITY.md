# Data Availability And Dataset Layout

This lightweight GitHub repository does not include the cleaned image files. It includes the manifests and audit records needed to verify the separately provided datasets.

The SHA256-cleaned image datasets used in the manuscript should be placed under:

```text
data/images_clean/
```

Expected datasets:

| Dataset | Images | Approx. local size | Public source |
|---|---:|---:|---|
| `F_new` | 13,950 | 0.180 GB | PlantVillage: https://github.com/spMohanty/PlantVillage-Dataset |
| `V_new` | 20,214 | 0.289 GB | PlantVillage: https://github.com/spMohanty/PlantVillage-Dataset |
| `M_new` | 4,842 | 0.412 GB | Mendeley Multi-Crop Leaf Disease Dataset: https://data.mendeley.com/datasets/z6jp232g5j |
| `M_new_drop5_drop7` | 4,440 | 0.342 GB | derived from `M_new`: https://data.mendeley.com/datasets/z6jp232g5j |
| `G_new` | 3,730 | 1.678 GB | Kaggle Plant Leaf Disease Recognition Dataset: https://www.kaggle.com/datasets/truongdinhit/plant-leaf-disease-recognition-dataset |

The per-dataset summary is archived at:

```text
data/images_clean_dataset_summary.csv
```

The per-image SHA256 manifest is archived at:

```text
data/images_clean_file_manifest_sha256.csv
```

The manuscript experiments used flat image directories with labels encoded in filenames:

```text
<class_id>_<image_id>.jpg
```

Main datasets:

- `F_new`: fruit disease subset derived from PlantVillage-style disease classes.
- `V_new`: vegetable disease subset derived from PlantVillage-style disease classes.
- `M_new`: complete 11-class Multi-Crop Leaf Disease Dataset Corn, Potato, Rice derived dataset.
- `M_new_drop5_drop7`: pre-arranged 9-class multi-crop disease subset used as the quality-control subset.
- `G_new`: auxiliary cross-crop visual category dataset, not treated as a main disease-recognition dataset.

The `G_new` source layout corresponds to `Data for Leaf Disease/Background Removed` in the Kaggle dataset. A related Mendeley Data page for Plant Leaf Disease Recognition Dataset is https://data.mendeley.com/datasets/5g238dv4ht.

The exact dataset inclusion after SHA256 de-duplication is recorded in:

```text
data_audit/duplicate_audit/exact_clean_keep_names.json
external_baselines/dinov2/audit/exact_clean_keep_names_external.json
```

pHash outputs are provided only as near-duplicate risk-audit records. pHash was not used to delete images and did not participate in feature extraction, clustering, rejection, or evaluation.

See `MANUAL_ASSETS.md` for how to restore the image folders before rerunning the experiments.
