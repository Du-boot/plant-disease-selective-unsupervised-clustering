# Weights And Checkpoints

## ConvNeXt Checkpoint

The main ConvNeXt experiments used a local external checkpoint:

```text
/data1/D/deploy/cn/model.safetensors
```

It is included in this local reproducibility archive:

```text
models/model.safetensors
```

It is approximately 1.5 GB, so it must be tracked with Git LFS when uploading to GitHub.

Recorded server file information:

```text
SHA256: 72b257ce7a079089c1bac54151807caf1b10d33a570fa9738a3ba437d24fc4d9
Size:   approximately 1.5 GB
Archive path: models/model.safetensors
Server path:  /data1/D/deploy/cn/model.safetensors
```

Until the checkpoint source is fully confirmed, the manuscript should not claim that ConvNeXt is intrinsically superior to DINOv2. The safe wording is that the ConvNeXt local-checkpoint setting achieved higher results in these experiments, while DINOv2 provides a fully public self-supervised external baseline.

## DINOv2 Checkpoint

The DINOv2 external baseline used:

```text
timm vit_base_patch14_dinov2
```

The public self-supervised DINOv2 checkpoint is included in this local reproducibility archive:

```text
models/dinov2_vit_base_patch14_lvd142m.safetensors
```

Recorded file information:

```text
SHA256: 55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b
Size:   346,334,872 bytes
Source: HuggingFace/timm cache for timm/vit_base_patch14_dinov2.lvd142m
```

The archived `code/external_dinov2_baselines.py` script loads this local checkpoint by default and resamples the absolute position embedding for 224 x 224 input.

If the local checkpoint is absent, the model can also be downloaded by `timm`. On the original server, HuggingFace downloads used:

```bash
HF_ENDPOINT=https://hf-mirror.com
```
