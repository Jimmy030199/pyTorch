import torch
from torch import nn, optim
from step1_model import Model
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# 讀取 CSV 檔案
df = pd.read_csv("taxi_fare_training.csv")

# 顯示整張表格
# print(df)

# 顯示某一欄（distance_km）
# print(df["distance_km"])
# 讀出並轉成 numpy 陣列 再轉成 torch tensor .view(-1, 1)變成二維 一筆資料只有一個特徵
x_true =torch.tensor(df["distance_km"].to_numpy(),dtype=torch.float32).view(-1, 1)
y_true =  torch.tensor(df["fare_ntd"].to_numpy(),dtype=torch.float32).view(-1, 1)


# 引入 model 
net = Model()
# 設定Optimizer（優化器）根據 loss 的梯度 來更新參數
opt = optim.SGD(net.parameters(),lr=0.01)
# 設定損失函式物件（loss function object）
loss_f = nn.MSELoss()

loss_hist=[]

for epoch in range(50):
    y_hat =net(x_true)
    # loss 是用 125 筆的平均誤差 算出來的
    loss = loss_f(y_true,y_hat)

    opt.zero_grad()
    # 算梯度: 對 loss 的偏微分（∂loss/∂參數） → 也叫做 梯度（gradient）
    loss.backward()
    opt.step()
    loss_hist.append(float(loss.item()))

    print(f'epoch:{epoch},loss:{loss.item()}')

OUT_DIR ="TRY_HW_results"
os.makedirs(OUT_DIR, exist_ok=True)

plt.figure()
plt.plot(range(len(loss_hist)),loss_hist,marker='o')

plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss')

plt.tight_layout()

loss_path = os.path.join(OUT_DIR,'loss.png')
plt.savefig(loss_path,dpi=150)

# 如果你不關閉圖表，下一次畫圖會疊在舊圖上，或記憶體一直累積。
plt.close()


net.eval() # 切到推論模式
with torch.no_grad(): # 關閉梯度追蹤
    y_hat = net(x_true)     # 預測

# 第一步：把 PyTorch tensor 轉成 NumPy 陣列

# detach()：切斷這個 tensor 和計算圖的連結（不再追蹤梯度）

# cpu()：確保 tensor 在 CPU 上（不能直接從 GPU tensor 轉 NumPy）

# .numpy()：轉成 NumPy 陣列（方便之後用 NumPy 或畫圖處理）

# 第二步：用 np.hstack 水平堆疊多個陣列

# np.hstack([...]) 是 水平（橫向）合併 多個陣列

# 要求這些陣列的「列數（row 數）」相同

#存csv用 
arr = np.hstack([
    x_true.detach().cpu().numpy(),
    y_true.detach().cpu().numpy(),
    y_hat.detach().cpu().numpy(),

])
pred_csv_path =os.path.join(OUT_DIR,'predictions.csv')
np.savetxt(pred_csv_path,arr,delimiter='-',comments='')


idx =x_true.squeeze(1).argsort()

x_sorted = x_true[idx]
yhat_sorted =y_hat[idx]

plt.figure()
plt.plot(x_true.detach().cpu().numpy(),y_true.detach().cpu().numpy(),'o',alpha=0.6,label='Date(x,y)')
plt.plot(x_true.detach().cpu().numpy(),y_hat.detach().cpu().numpy(),'x',alpha=0.8,label='Model output on training x')
plt.plot(x_sorted.detach().cpu().numpy(),yhat_sorted.detach().cpu().numpy(),'-',linewidth=2,label='Connected model outputs')

plt.xlabel('x')
plt.ylabel('value')
plt.title('Model Outputs on Training Points')
plt.legend()
plt.tight_layout()
scatter_outputs_path =os.path.join(OUT_DIR,'data_with_true_model_outputs.png')
plt.savefig(scatter_outputs_path,dpi=150)
plt.close()



