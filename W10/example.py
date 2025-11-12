# -*- coding: utf-8 -*-
"""
Example: Scaled Dot-Product Attention (for multi-head)
-----------------------------------------------------
此程式示範：
  - Q, K, V 各為 [BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_K/D_V]
  - 實作注意力核心公式：
        Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
  - 最後印出各張量形狀以確認維度一致性
"""

import torch
import torch.nn.functional as F
import math

# === 定義 Scaled Dot-Product Attention ===
def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)  # 每個 head 的維度
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output, attn_weights


# === 參數設定 ===
BATCH_SIZE = 4
NUM_HEADS = 8
SEQ_LEN = 10
D_K = 64
D_V = 64

# === 建立假資料 ===
q = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_K)
k = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_K)
v = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_V)

# === 執行 Attention ===
output, weights = scaled_dot_product_attention(q, k, v)

# === 印出結果形狀 ===
print(f"Q shape: {q.shape}")
print(f"K shape: {k.shape}")
print(f"V shape: {v.shape}")
print("-" * 40)
print(f"Output shape:  {output.shape}")   # [BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_V]
print(f"Weights shape: {weights.shape}")  # [BATCH_SIZE, NUM_HEADS, SEQ_LEN, SEQ_LEN]
