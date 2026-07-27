# Data Availability And Dataset Layout

The manuscript uses SHA256-cleaned image folders. The full image datasets are
large and are provided separately:

[Baidu Netdisk package](https://pan.baidu.com/s/1d8sLfgtz79Fjs-s7zvm2YQ?pwd=1234)

Extraction code: `1234`

After downloading, place the four final folders under:

```text
data/images_clean/F_new/
data/images_clean/V_new/
data/images_clean/M_new/
data/images_clean/G_new/
```

Final V13 datasets:

| Folder | Manuscript name | Role |
|---|---|---|
| `F_new` | PV-Fruit | PlantVillage fruit disease subset |
| `V_new` | PV-Vegetable | PlantVillage vegetable disease subset |
| `M_new` | MCLD-11 | Complete 11-class multi-crop disease dataset after SHA256 de-duplication |
| `G_new` | DFLD-BR-4 | Auxiliary de-background crop-category dataset |

The previous `M_new_drop5_drop7` / MCLD-9 subset is retained only in historical
notes and is not part of the final V13 main experiment.

The per-dataset summary and per-image SHA256 manifest are archived at:

```text
data/images_clean_dataset_summary.csv
data/images_clean_file_manifest_sha256.csv
```

The manuscript data protocol is:

- SHA256 is used to identify and remove byte-level exact duplicate images.
- pHash is used only for near-duplicate risk audit.
- pHash results do not delete images and do not participate in feature
  extraction, clustering, rejection, or evaluation.

The cleaned datasets use flat image directories with labels encoded in filenames:

```text
<class_id>_<image_id>.jpg
```
