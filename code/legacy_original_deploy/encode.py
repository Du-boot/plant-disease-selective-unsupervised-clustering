# encode.py —— 离线用 ConvNeXt-XL(in22k) 提特征（本地 .safetensors 权重）
import os, glob
import numpy as np
from PIL import Image
import torch
import timm
from torchvision import transforms
from safetensors.torch import load_file
from tqdm import tqdm
# === 新增：全链路可复现设置 ===
import random
def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # 或 ":4096:2"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def main():
    set_seed(42)  # 新增：固定随机源
    os.environ["HF_HUB_OFFLINE"] = "1"  # 保险：禁用 HF 联网
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda available?", torch.cuda.is_available())

    # 1) 构建模型（不走在线预训练，也不要传 checkpoint_path）
    model = timm.create_model(
        "convnext_xlarge_in22k",
        pretrained=False,
        num_classes=0,
        global_pool="avg",
    )

    # 2) 从本地 .safetensors 加载权重（手动 load_state_dict）
    CKPT = "/data1/D/deploy/cn/model.safetensors"  # ← 换成你的实际路径
    # CKPT = "/data1/dukewei/deploy5/convnext_pth/convnext_xlarge_22k_224.pth"  # ← 换成你的实际路径
    sd = load_file(CKPT, device="cpu")
    new_sd = {}
    for k, v in sd.items():
        nk = k[6:] if k.startswith("model.") else k   # 去掉可能的 "model." 前缀
        if nk.startswith(("head.weight", "head.bias")):
            continue                                   # 丢弃分类头
        new_sd[nk] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"[load] missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval().to(device)
    torch.set_grad_enabled(False)  # 新增：关闭梯度，避免隐式状态


    # 3) 预处理（ImageNet 标准化 —— 千万别少了这个）
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    # 4) 输入/输出（当前只扫描一层，不递归；按需改路径）
    train_path = "/data1/D/1_new/"   # ← 你的图片目录
    # train_path = "/data1/dukewei/2_new"   # ← 你的图片目录
    out_dir = "encode_2026071301"
    os.makedirs(out_dir, exist_ok=True)
    out_feat = os.path.join(out_dir, "1_2048.txt")
    out_name = os.path.join(out_dir, "1_2048_name.txt")
# ★ 新增：相对锚点（= train_path 的上一级目录）
    root_anchor = os.path.dirname(os.path.normpath(train_path))
    # 5) 遍历图像并导出 2048 维特征（路径 \t 向量<=>拼接 \t 标签）
    with open(out_feat, "w") as ffeat, open(out_name, "w") as fname:
        for p in tqdm(sorted(glob.glob(os.path.join(train_path, "*"))), ncols=80):
            try:
                label = os.path.basename(p).split("_")[0]
                img = Image.open(p).convert("RGB")
                x = tfm(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    vec = model(x).squeeze(0).cpu().numpy().astype("float32")
                    # ★ 改成写相对路径：2_new/0_1.jpg
                rel = os.path.normpath(os.path.relpath(p, root_anchor))
                ffeat.write(f"{rel}\t{'<=>'.join(f'{z:.6f}' for z in vec)}\t{label}\n")
                fname.write(f"{rel}\n")
                # ffeat.write(f"{p}\t{'<=>'.join(f'{z:.6f}' for z in vec)}\t{label}\n")
                # fname.write(f"{p}\n")
            except Exception as e:
                print(f"[WARN] 跳过 {p}: {e}")

if __name__ == "__main__":
    main()
