# Weights And Checkpoints

This lightweight GitHub repository does not include model checkpoints. Place the required files under `models/` before running feature extraction or DINOv2 baselines.

## ConvNeXt Checkpoint

The main ConvNeXt experiments used the public timm ConvNeXt-XLarge ImageNet-22K checkpoint, stored locally as:

```text
/data1/D/deploy/cn/model.safetensors
```

Required local archive path:

```text
models/model.safetensors
```

Recorded file information:

```text
SHA256: 72b257ce7a079089c1bac54151807caf1b10d33a570fa9738a3ba437d24fc4d9
Size:   approximately 1.5 GB
Archive path: models/model.safetensors
Server path:  /data1/D/deploy/cn/model.safetensors
```

The local `model.safetensors` file corresponds to the public timm ConvNeXt-XLarge ImageNet-22K checkpoint (`convnext_xlarge.fb_in22k`; legacy code name `convnext_xlarge_in22k`). The model was used as a frozen 2048-D feature extractor and was not fine-tuned on the manuscript datasets.

Public ConvNeXt-XLarge ImageNet-22K download references:

```text
Hugging Face timm model page:
https://huggingface.co/timm/convnext_xlarge.fb_in22k

Original Facebook/timm weight URL:
https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth
```

The manuscript archive stores the checkpoint in safetensors format. If downloading the original `.pth` file, convert or load it consistently with the archived extraction script before comparing feature outputs.

## DINOv2 Checkpoint

The DINOv2 external baseline used:

```text
timm vit_base_patch14_dinov2
```

Required local archive path:

```text
models/dinov2_vit_base_patch14_lvd142m.safetensors
```

Recorded file information:

```text
SHA256: 55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b
Size:   346,334,872 bytes
Source: HuggingFace/timm cache for timm/vit_base_patch14_dinov2.lvd142m
```

Public download page:

```text
https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m
```

The archived `code/external_dinov2_baselines.py` script loads this local checkpoint by default and resamples the absolute position embedding for 224 x 224 input.

If the local checkpoint is absent, the model can also be downloaded by `timm`. On the original server, HuggingFace downloads used:

```bash
HF_ENDPOINT=https://hf-mirror.com
```
