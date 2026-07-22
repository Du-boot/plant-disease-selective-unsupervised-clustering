# Data Availability And Dataset Layout

This lightweight GitHub repository does not include the cleaned image files. It includes the manifests and audit records needed to verify the separately provided datasets.

The SHA256-cleaned image datasets used in the manuscript should be placed under:

```text
data/images_clean/
```

Expected datasets:

| Dataset | Images | Approx. local size |
|---|---:|---:|
| `F_new` | 13,950 | 0.180 GB |
| `V_new` | 20,214 | 0.289 GB |
| `M_new` | 4,842 | 0.412 GB |
| `M_new_drop5_drop7` | 4,440 | 0.342 GB |
| `G_new` | 3,730 | 1.678 GB |

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

The exact dataset inclusion after SHA256 de-duplication is recorded in:

```text
data_audit/duplicate_audit/exact_clean_keep_names.json
external_baselines/dinov2/audit/exact_clean_keep_names_external.json
```

pHash outputs are provided only as near-duplicate risk-audit records. pHash was not used to delete images and did not participate in feature extraction, clustering, rejection, or evaluation.

See `MANUAL_ASSETS.md` for how to restore the image folders before rerunning the experiments.
