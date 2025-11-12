# ===============================================================
# scaled_dot_product_attention.py
# 實作 Scaled Dot-Product Attention (支援 Torch 與 Numpy)
# ===============================================================

import math
from typing import Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception as e:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------
# 1️⃣ Scaled Dot-Product Attention
# ---------------------------------------------------------------
def scaled_dot_product_attention(q, k, v, mask=None, dropout_p: float = 0.0):
    """
    q, k, v: (B, H, L, D)
    mask: (B, 1, L, L) 或 None
    回傳:
      output: (B, H, L, D)
      attn_weights: (B, H, L, L)
    """
    d_k = q.size(-1) if TORCH_AVAILABLE else q.shape[-1]

    if TORCH_AVAILABLE:
        # ---- Torch 版本 ----
        scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        if dropout_p > 0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)
        output = attn_weights @ v
        return output, attn_weights
    else:
        # ---- Numpy 版本 ----
        d_k=q.size[-1]
        scores = np.matmul(q, np.swapaxes(k, -2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        scores_max = scores.max(axis=-1, keepdims=True)
        exp = np.exp(scores - scores_max)
        attn_weights = exp / exp.sum(axis=-1, keepdims=True)
        output = np.matmul(attn_weights, v)
        return output, attn_weights


# ---------------------------------------------------------------
# 2️⃣ Padding Mask
# ---------------------------------------------------------------
def build_padding_mask(lengths, max_len):
    """
    根據每個 batch 的有效長度建立 padding mask
    """
    try:
        device = lengths.device if hasattr(lengths, "device") else "cpu"
        rng = torch.arange(max_len, device=device).unsqueeze(0)
        mask = (rng < lengths.unsqueeze(1)).to(torch.bool)
        return mask.unsqueeze(1).unsqueeze(1)
    except Exception:
        lengths = np.asarray(lengths)
        rng = np.arange(max_len)[None, :]
        mask = (rng < lengths[:, None])
        return mask[:, None, None, :]


# ---------------------------------------------------------------
# 3️⃣ Look-Ahead Mask
# ---------------------------------------------------------------
def build_look_ahead_mask(seq_len):
    """
    建立下三角矩陣 mask，防止看到未來資訊
    """
    try:
        return torch.tril(torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool))
    except Exception:
        m = np.tril(np.ones((seq_len, seq_len), dtype=bool))
        return m[None, None, :, :]


# ---------------------------------------------------------------
# 4️⃣ 主程式測試
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("TORCH_AVAILABLE =", TORCH_AVAILABLE)

    try:
        torch.manual_seed(7)
        B, H, L, D = 2, 2, 5, 4

        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)
        v = torch.randn(B, H, L, D)

        # ---- 無 mask ----
        out, attn = scaled_dot_product_attention(q, k, v, mask=None)
        print("[No Mask] output shape:", tuple(out.shape))
        print("[No Mask] attn shape:", tuple(attn.shape))
        print("Row sums (應該≈1):", attn[0, 0, 0].sum().item())

        # ---- Padding mask ----
        lengths = torch.tensor([3, 5])
        pad_mask = build_padding_mask(lengths, max_len=L)
        out_padded, attn_padded = scaled_dot_product_attention(q, k, v, mask=pad_mask)
        print("Check pad columns (index>=3) 是否≈0?", attn_padded[0, 0, 3:].sum().item())

        # ---- Look-ahead mask ----
        look_mask = build_look_ahead_mask(L)
        la_mask = look_mask.expand(B, H, L, L)
        out_la, attn_la = scaled_dot_product_attention(q, k, v, mask=la_mask)
        print("Look-ahead Mask 是否為下三角 (印第一列):", attn_la[0, 0, 0])

        # ---- Softmax 驗證 ----
        scores = torch.tensor([
            [1.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0]
        ])
        weights = torch.softmax(scores, dim=-1)
        print("3x3 softmax:\n", weights)

    except Exception as e:
        # ---- Numpy fallback ----
        np.random.seed(7)
        B, H, L, D = 2, 2, 5, 4
        q = np.random.randn(B, H, L, D)
        k = np.random.randn(B, H, L, D)
        v = np.random.randn(B, H, L, D)

        out, attn = scaled_dot_product_attention(q, k, v, mask=None)
        print("[NP] output shape:", out.shape)
        print("Row sums (應≈1):", attn[0, 0, 0].sum())

        scores = np.array([
            [1.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0]
        ])
        exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights = exp / exp.sum(axis=-1, keepdims=True)
        print("[NP] 3x3 softmax:\n", weights)
