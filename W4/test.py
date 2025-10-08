# -*- coding: utf-8 -*-
"""
test.py
載入訓練好的模型，隨機挑選幾張 Fashion-MNIST 圖片做預測並繪出結果
"""

import os
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from cnn import CNN   # ← 你的 CNN 模型類別
from matplotlib import font_manager

# ========== 可調參數 ==========
DATA_DIR = "data"
OUT_DIR = "result"
CKPT_PATH = Path(OUT_DIR) / "best_cnn.pth"  # 模型檔路徑
ROWS, COLS = 2, 5                            # 輸出圖片網格 (共 ROWS×COLS 張)
SEED = 42
MEAN, STD = 0.2861, 0.3530                   # Fashion-MNIST 正規化參數
FIGSIZE = (16, 8)
TITLE_FZ = 11

# Windows 下常見中文字體（用來顯示中文標題）
ZH_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msjh.ttc",
    r"C:\Windows\Fonts\msjhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]

# 類別名稱
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# ========== 輔助函式 ==========
def set_seed(seed):
    """固定隨機種子，確保結果可重現"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_zh_font():
    """設定中文字體 (避免 matplotlib 顯示亂碼)"""
    for p in ZH_FONT_CANDIDATES:
        if Path(p).exists():
            plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=p).get_name()]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[INFO] 已啟用中文字體: {p}")
            return
    print("[WARN] 找不到中文字體，中文可能無法正常顯示。")


def denorm_img(x):
    """把 Normalize 後的圖片轉回 0~1 區間方便顯示"""
    x = x * STD + MEAN
    return x.clamp(0, 1)


# ========== 主程式 ==========
def main():
    set_seed(SEED)
    set_zh_font()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 載入測試資料集 ---
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,))
    ])
    test_set = datasets.FashionMNIST(root=DATA_DIR, train=False, download=True, transform=tfm)

    # --- 載入模型 ---
    model = CNN(num_classes=10).to(device)
    if CKPT_PATH.exists():
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"[INFO] 已載入模型權重: {CKPT_PATH}")
    else:
        print(f"[WARN] 模型檔案不存在: {CKPT_PATH}，使用隨機初始化權重！")

    model.eval()

    # --- 隨機挑選 N 張圖片 ---
    N = ROWS * COLS
    indices = list(range(len(test_set)))
    random.shuffle(indices)
    indices = indices[:N]

    images, gts, preds, confs = [], [], [], []

    with torch.no_grad():
        for idx in indices:
            img_t, label = test_set[idx]
            logits = model(img_t.unsqueeze(0).to(device))  # [1, 10]
            prob = F.softmax(logits, dim=1)[0]
            pred_id = int(torch.argmax(prob).item())
            conf = float(prob[pred_id].item())

            images.append(img_t.cpu())
            gts.append(label)
            preds.append(pred_id)
            confs.append(conf)

    # --- 畫出預測結果 ---
    os.makedirs(OUT_DIR, exist_ok=True)
    fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)

    for i in range(N):
        ax = fig.add_subplot(ROWS, COLS, i + 1)
        img = denorm_img(images[i])
        ax.imshow(img.squeeze(), cmap="gray", interpolation="nearest")
        ax.axis("off")

        gt_name = CLASS_NAMES[gts[i]]
        pred_name = CLASS_NAMES[preds[i]]
        conf_txt = f"{confs[i]:.3f}"

        # 標題：真實 vs 預測 + 信心值
        title = f"真: {gt_name}\n預: {pred_name} ({conf_txt})"
        ax.set_title(title, color="green", fontsize=TITLE_FZ, pad=6)

    fig.suptitle("Fashion-MNIST 測試集預測結果", fontsize=16, y=0.98)

    out_path = Path(OUT_DIR) / "test_grid.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 圖片已輸出: {out_path.resolve()}")


if __name__ == "__main__":
    main()
