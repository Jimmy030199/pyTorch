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
# 讀出並轉成 numpy 陣列 再轉成 torch tensor
x_true =torch.tensor(df["distance_km"].to_numpy(),dtype=torch.float32)
y_true =  torch.tensor(df["fare_ntd"].to_numpy(),dtype=torch.float32)


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