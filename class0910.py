import torch
from torch import nn, optim
from step1_model import Model
import matplotlib.pyplot as plt
import numpy as np
import os

torch.manual_seed(0)
n = 100

# (n,1) = 產生的矩陣大小
# n = 行數 (rows)
# 1 = 列數 (columns)
# torch.randn 會回傳tensor的格式
# 是二維的
x = torch.randn(n,1)

# torch.tensor(10.0) = 純量 (scalar)
# torch.tensor([10.0]) = 向量 (vector)，長度為 1

w_true = torch.tensor([10.0]) 
b_true = torch.tensor([3.0])

# w_true 是 (1,)
# x 是 (n,1)
# 用 * → 逐元素相乘，不是矩陣乘法。
# PyTorch 自動把 (1,) broadcast 成 (n,1)，所以相乘後還是 (n,1)。
# 逐元素相乘 (element-wise, *)

# 規則：形狀必須相同，或是能透過 broadcasting 對齊。
# ✅ 同維度、同形狀可以直接相乘
# a = torch.tensor([[1,2],[3,4]])
# b = torch.tensor([[10,20],[30,40]])
# print(a * b)   # shape (2,2)
# 結果：
# tensor([[10, 40],
#         [90,160]])
y = w_true * x + b_true + torch.randn(n,1) * 2.5

net = Model()

# SGD 每筆資料更新一次 100筆共100次
opt = optim.SGD(net.parameters(),lr=0.1)

# 表示 建立一個均方誤差損失函數，用來計算模型預測值與真實值的平均平方差，是迴歸任務常用的 loss function。
loss_f = nn.MSELoss()

loss_hist=[]

for epoch in range(11):
    # net 是你的模型 (繼承 nn.Module)
    # x 是輸入資料 (features)
    # 執行 net(x) 會呼叫模型的 forward，得到模型的 預測值 y_hat
    y_hat =net(x)

    # 這一步計算「預測和真實的差距」，回傳一個 標量 (scalar tensor)
    loss = loss_f(y,y_hat)

    # 在 PyTorch 中，梯度會「累加」(accumulate)，不是自動覆蓋
    # 如果不先清掉，新的梯度會加在舊的梯度上，造成錯誤更新
    # 所以每次反向傳播前，都要執行這一步
    opt.zero_grad()

    # PyTorch 根據 loss 的 計算圖 (computational graph)，自動算出：
    #     也就是所有參數的偏微分 (gradient)
    # 算出的梯度會存到每個參數的 .grad 屬性裡
    # for p in net.parameters():
    #     print(p.grad)  # 這裡就是梯度
    loss.backward()

    #     更新參數

    # Optimizer（這裡是 torch.optim.SGD）會根據公式更新權重：
    # w:=w−η⋅∇w​L
    opt.step()

    loss_hist.append(float(loss.item()))
 
    # print(f'epoch:{epoch},loss:{loss.item()}')
    # print(loss)

OUT_DIR ="results"
os.makedirs(OUT_DIR,exist_ok=True)

plt.figure()
plt.plot(range(len(loss_hist)),loss_hist,marker='o')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss')
plt.grid(True,linestyle="--",alpha=0.5)
plt.tight_layout()

# 用來拼接路徑
loss_path = os.path.join(OUT_DIR,'loss.png')

plt.savefig(loss_path,dpi=150)
plt.close()

# gpu暫停 不更新資料 讓cpu執行 把模型 net 切換成 評估模式 (evaluation mode)
net.eval() 

# 在 Python 裡，with 語句用來 管理資源，常見例子是開檔案：
with torch.no_grad():
    #     把輸入 x 丟進模型 net，得到預測值 y_hat
    # 因為包在 torch.no_grad() 裡，所以這個計算 不會被 autograd 記錄，節省資源
    # PyTorch 的 autograd 就是「自動微分系統」，透過計算圖記錄運算流程，讓你只要呼叫 .backward()，
    # 就能得到所有需要的梯度，用來更新模型參數。
    y_hat = net(x)

# 1. x.detach()
# detach()：把 tensor 從 計算圖 (computational graph) 分離出來
# 因為在訓練時，x、y、y_hat 可能有 requires_grad=True
# 如果你只是要輸出或存檔，不需要梯度 → 用 detach() 切斷 autograd 追蹤

# 2. .cpu()
# 把 tensor 從 GPU 移回 CPU
# 因為 NumPy 只能處理 CPU tensor，不能直接用 GPU tensor

# 3. .numpy()
# 把 PyTorch tensor 轉成 NumPy array
# 這樣就能用 NumPy 的函式處理（如 hstack、存檔、繪圖）

# np.hstack（horizontal stack，水平拼接）

arr =np.hstack([
    x.detach().cpu().numpy(),
    y.detach().cpu().numpy(),
    y_hat.detach().cpu().numpy(),

])



pred_csv_path =os.path.join(OUT_DIR,'predictions.csv')
np.savetxt(pred_csv_path,arr,delimiter='-',comments='')

# 如果那一維 = 1 → 移除
# 如果那一維 ≠ 1 → 不動
# argsort() 會回傳「從小到大排序時的索引值」
# 畫圖的差別
# 沒排序：
# x = [3,1,2]，折線會亂跳：3 → 1 → 2
# 排序後：
# x_sorted = [1,2,3]，折線會平順：1 → 2 → 3
idx =x.squeeze(1).argsort()
x_sorted = x[idx]
yhat_sorted =y_hat[idx]

plt.figure()
plt.plot(x.detach().cpu().numpy(),y.detach().cpu().numpy(),'o',alpha=0.6,label='Date(x,y)')
plt.plot(x.detach().cpu().numpy(),y_hat.detach().cpu().numpy(),'x',alpha=0.8,label='Model output on training x')
plt.plot(x_sorted.detach().cpu().numpy(),yhat_sorted.detach().cpu().numpy(),'-',linewidth=2,label='Connected model outputs')

plt.xlabel('x')
plt.ylabel('value')
plt.title('Model Outputs on Training Points')
plt.legend()
plt.tight_layout()
scatter_outputs_path =os.path.join(OUT_DIR,'data_with_true_model_outputs.png')
plt.savefig(scatter_outputs_path,dpi=150)
plt.close()