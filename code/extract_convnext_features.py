#!/usr/bin/env python3
"""Extract frozen ConvNeXt features for the archived image datasets.

The output layout matches the feature_root expected by
`protocolA_followup_experiments.py`:

    <feature-root>/features/<dataset>.npy
    <feature-root>/features/<dataset>.names.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import timm
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_DATASETS = ["F_new", "V_new", "M_new_drop5_drop7", "G_new", "M_new"]


def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ImageDataset(Dataset):
    def __init__(self, root: Path, names: list[str], tfm):
        self.root = root
        self.names = names
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        image = Image.open(self.root / name).convert("RGB")
        return self.tfm(image), name


def image_names(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def build_model(checkpoint: Path, device: torch.device):
    model = timm.create_model(
        "convnext_xlarge_in22k",
        pretrained=False,
        num_classes=0,
        global_pool="avg",
    )
    state = load_file(str(checkpoint), device="cpu")
    cleaned = {}
    for key, value in state.items():
        new_key = key[6:] if key.startswith("model.") else key
        if new_key.startswith(("head.weight", "head.bias")):
            continue
        cleaned[new_key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[load] missing={len(missing)}, unexpected={len(unexpected)}", flush=True)
    model.eval().to(device)
    return model


def extract_dataset(model, device: torch.device, dataset_root: Path, out_feat: Path, out_names: Path, batch_size: int) -> None:
    if out_feat.exists() and out_names.exists():
        print(f"[skip] cached {out_feat}", flush=True)
        return
    names = image_names(dataset_root)
    if not names:
        raise FileNotFoundError(f"No images found in {dataset_root}")
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    loader = DataLoader(
        ImageDataset(dataset_root, names, tfm),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    chunks = []
    seen = []
    torch.set_grad_enabled(False)
    for step, (images, batch_names) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            vec = model(images).detach().cpu().numpy().astype("float32")
        chunks.append(vec)
        seen.extend(list(batch_names))
        if step % 20 == 0:
            print(f"[extract] {dataset_root.name}: batch {step}/{len(loader)}", flush=True)
    features = np.concatenate(chunks, axis=0)
    out_feat.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_feat, features)
    out_names.write_text(json.dumps(seen, ensure_ascii=False), encoding="utf-8")
    print(f"[save] {dataset_root.name}: {features.shape} -> {out_feat}", flush=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=repo_root / "data" / "images_clean")
    parser.add_argument("--checkpoint", type=Path, default=repo_root / "models" / "model.safetensors")
    parser.add_argument("--feature-root", type=Path, default=repo_root / "outputs" / "convnext_feature_root")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)
    model = build_model(args.checkpoint, device)
    for dataset in args.datasets:
        dataset_root = args.data_root / dataset
        extract_dataset(
            model,
            device,
            dataset_root,
            args.feature_root / "features" / f"{dataset}.npy",
            args.feature_root / "features" / f"{dataset}.names.json",
            args.batch_size,
        )


if __name__ == "__main__":
    main()
