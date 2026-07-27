# Weights And Checkpoints

The full checkpoint files are large and are provided in the same external asset
package as the cleaned datasets:

[Baidu Netdisk package](https://pan.baidu.com/s/1d8sLfgtz79Fjs-s7zvm2YQ?pwd=1234)

Extraction code: `1234`

Expected paths after unpacking:

```text
models/model.safetensors
models/dinov2_vit_base_patch14_lvd142m.safetensors
```

## ConvNeXt Checkpoint

The main manuscript pipeline uses:

```text
timm model name: convnext_xlarge_in22k
checkpoint path: models/model.safetensors
input size: 224 x 224
feature output: 2048-D global pooled feature
training in this study: none; all parameters frozen
```

This checkpoint corresponds to the public timm ConvNeXt-XLarge ImageNet-22K
weight used as a frozen visual feature extractor.

Recorded file information from the local reproducibility archive:

```text
SHA256: 72b257ce7a079089c1bac54151807caf1b10d33a570fa9738a3ba437d24fc4d9
Size:   1,571,635,946 bytes
```

## DINOv2 Checkpoint

The external self-supervised baseline uses:

```text
timm model name: vit_base_patch14_dinov2
checkpoint path: models/dinov2_vit_base_patch14_lvd142m.safetensors
input size: 224 x 224
training in this study: none; all parameters frozen
```

Recorded file information:

```text
SHA256: 55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b
Size:   346,334,872 bytes
Source: HuggingFace/timm cache for timm/vit_base_patch14_dinov2.lvd142m
```

If the local DINOv2 checkpoint is absent, `timm` can download the public model
online. On the original server, HuggingFace downloads used:

```bash
HF_ENDPOINT=https://hf-mirror.com
```
