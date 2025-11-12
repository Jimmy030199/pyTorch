# verify_lookahead.py
import math
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    q,k,v: (B,H,L,D), mask: (B,1,L,L) or (1,1,L,L) with True=allowed / False=blocked
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # 0=禁止 → -inf
    attn = F.softmax(scores, dim=-1)
    out = attn @ v
    return out, attn

def build_look_ahead_mask(L: int):
    # 下三角為 1（允許），上三角為 0（禁止）
    return torch.tril(torch.ones((1, 1, L, L), dtype=torch.bool))

def build_look_ahead_mask(L):
    """建立 Look-ahead 下三角矩陣"""
    mask = torch.tril(torch.ones((L, L), dtype=torch.int))
    return mask

if __name__ == "__main__":
    torch.manual_seed(7)
    B, H, L, D = 1, 1, 5, 4

    # 產生固定的 q/k/v 方便重現
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)

    # 建立 Look-ahead mask 並套用
    la_mask = build_look_ahead_mask(L)           # (1,1,L,L)
    out, attn = scaled_dot_product_attention(q, k, v, mask=la_mask)

    # ==== 驗證 1：上三角（未來位置）為 0 ====
    # 取出注意力矩陣 (L,L)，把對角線以上的最大值抓出來
    A = attn[0, 0]                                # (L,L)
    upper = torch.triu(A, diagonal=1)            # 上三角（不含對角）
    print("① 上三角最大值（應≈0）:", float(upper.max()))
    # 也可硬性檢查（容忍浮點誤差）
    assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-6), "未來位置沒有完全被遮！"

    # ==== 驗證 2：第 1 個 token 只能看自己 ====
    row0 = A[0]                                   # (L,)
    print("② attn_weights[0,0,0] 向量：", row0.tolist())
    print("   非零索引：", [i for i,v in enumerate(row0) if v > 1e-9])
    # 只剩 index=0 應為非零
    assert (row0[0] > 1e-9) and torch.all(row0[1:] < 1e-9), "第1個 token 卻能看到未來！"

    # ==== 驗證 3：第 3 個 token 只能看前 3 個 ====
    row2 = A[2]                                   # (L,)
    print("③ attn_weights[0,0,2] 向量：", row2.tolist())
    print("   非零索引：", [i for i,v in enumerate(row2) if v > 1e-9])
    # 只允許 0,1,2 位置非零
    assert torch.all(row2[:3] > 1e-9) and torch.all(row2[3:] < 1e-9), "第3個 token 看到了未來！"

    # 其他 sanity check：每列總和≈1
    rowsum = A.sum(-1)
    print("④ 每列總和（應≈1）：", rowsum.tolist())
    assert torch.allclose(rowsum, torch.ones_like(rowsum), atol=1e-6)

    print("\n✅ 驗證通過：Look-ahead Mask 正常阻擋未來位置。")


   


    
    L = 5  # 序列長度，可自行改
    mask = build_look_ahead_mask(L)
    print("🔻 Look-ahead Mask 下三角矩陣 (1=允許看, 0=遮住):\n")
    print(mask)

    

