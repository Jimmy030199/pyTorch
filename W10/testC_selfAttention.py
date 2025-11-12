# -*- coding: utf-8 -*-
"""
實驗 C：Self-Attention 點積機制觀察
目的：
  - 理解 scaled dot-product attention 如何根據 q/k 內積計算權重。
  - 我們讓第 0 個 token 的 query 特別接近第 0 個 key，觀察注意力分佈。

輸出：
  - 顯示最後一列（對應最後一個 token）對所有 key 的注意力權重。
"""

import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 或 SimHei / Noto Sans CJK
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示錯誤


# === 定義 scaled dot-product attention ===
def sdp_attention(q, k, v, mask=None):
    d = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)  # QK^T / sqrt(d)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    w = F.softmax(scores, dim=-1)                    # softmax → 注意力權重
    out = w @ v
    return out, w


# === 基本設定 ===
L, D = 6, 2  # 序列長度 L=6，每個 token 維度 D=2
q = torch.zeros(1, 1, L, D)
k = torch.zeros(1, 1, L, D)
v = torch.arange(L).float().view(1, 1, L, 1).repeat(1, 1, 1, 1)  # value 即位置編號

# === 讓最後一個 token 的 q/k 特別與第 0 個 token 相似 ===
q[0, 0, 0] = torch.tensor([1.0, 0.0])  # query of token 0
k[0, 0, 0] = torch.tensor([1.0, 0.0])  # key   of token 0

# === 其他位置的 q/k 幾乎無關 ===
for i in range(1, L):
    q[0, 0, i] = torch.tensor([0.0, 1.0])
    k[0, 0, i] = torch.tensor([0.0, 1.0])

# === 執行 Attention ===
out, attn = sdp_attention(q, k, v)

# === 印出注意力權重 ===
print("最後一列注意力權重（應該在位置 0 最高）:")
print(attn[0, 0, -1])  # 最後一個 token 對所有 key 的注意力分佈

# === 視覺化 ===
plt.figure(figsize=(6, 3))
plt.bar(range(L), attn[0, 0, -1].detach().numpy())
plt.title("最後一個 token 對各 key 的注意力權重")
plt.xlabel("Key index")
plt.ylabel("Attention weight")
plt.grid(True)
plt.show()
