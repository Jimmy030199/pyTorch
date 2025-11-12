# -*- coding: utf-8 -*-
"""
conv1d_receptive_field_demo.py
說明：
  - 建立多層 Conv1d，每層 kernel_size=3, stride=1, padding=1
  - 權重全為 1（方便觀察感受野擴散）
  - 在序列中心放一個脈衝 (one-hot)
  - 輸出觀察非零範圍寬度與理論感受野比較
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# === 參數設定 ===
L = 21                 # 序列長度
num_layers = 3         # 卷積層數
kernel_size = 3
stride = 1
padding = 1

# === 建立輸入：中心為 1 其餘為 0 ===
x = torch.zeros(1, 1, L)
x[0, 0, L // 2] = 1.0  # 中間放 one-hot 脈衝

# === 建立多層 Conv1d ===
layers = []
for i in range(num_layers):
    conv = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    nn.init.ones_(conv.weight)  # 權重全部設為 1
    layers.append(conv)

net = nn.Sequential(*layers)

# === 前向傳遞（不計梯度） ===
with torch.no_grad():
    y = net(x)

# === 找出非零區間 ===
nz = (y[0, 0] != 0).nonzero().flatten()
left, right = nz[0].item(), nz[-1].item()
nonzero_width = right - left + 1

# === 理論感受野 ===
receptive_field = 1 + (kernel_size - 1) * num_layers

# === 輸出結果 ===
print(f"非零區間寬度 = {nonzero_width}")
print(f"理論感受野 R_{num_layers} = {receptive_field}")

# === 視覺化 ===
plt.figure(figsize=(8, 3))
plt.plot(y[0, 0].numpy(), 'o-', label=f'Output after {num_layers} Conv1d layers')
plt.axvline(left, color='red', linestyle='--', label='Left boundary')
plt.axvline(right, color='red', linestyle='--', label='Right boundary')
plt.title(f"Non-zero width = {nonzero_width}, Theoretical RF = {receptive_field}")
plt.xlabel("Position index")
plt.ylabel("Activation")
plt.legend()
plt.grid(True)
plt.show()
