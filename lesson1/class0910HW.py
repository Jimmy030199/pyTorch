import torch
from torch import nn, optim
from step1_model import Model
import matplotlib.pyplot as plt
import numpy as np
import os

import csv

csv_file_path = r'C:\Users\User\Desktop\pyTorch\lesson1\taxi_fare_training.csv'

data = np.genfromtxt(csv_file_path,delimiter=',',skip_header=1)
print(data)
x =torch.tensor(data[:,0],dtype=torch.float32).view(-1,1)
y =torch.tensor(data[:,1],dtype=torch.float32).view(-1,1)




net = Model()
opt = optim.SGD(net.parameters(),lr=0.01)
loss_f = nn.MSELoss()

loss_hist=[]

for epoch in range(50):
    y_hat =net(x)
    loss = loss_f(y,y_hat)
    opt.zero_grad()
    loss.backward()
    opt.step()

    loss_hist.append(float(loss.item()))
 
    # print(f'epoch:{epoch},loss:{loss.item()}')
    # print(loss)

OUT_DIR ="HW_results"
os.makedirs(OUT_DIR,exist_ok=True)

plt.figure()
plt.plot(range(len(loss_hist)),loss_hist,marker='o')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss')
plt.grid(True,linestyle="--",alpha=0.5)
plt.tight_layout()
loss_path = os.path.join(OUT_DIR,'loss.png')
plt.savefig(loss_path,dpi=150)
plt.close()

# gpu暫停 不更新資料 讓cpu執行
net.eval() 
with torch.no_grad():
    y_hat = net(x)

arr =np.hstack([
    x.detach().cpu().numpy(),
    y.detach().cpu().numpy(),
    y_hat.detach().cpu().numpy(),

])

pred_csv_path =os.path.join(OUT_DIR,'predictions.csv')
np.savetxt(pred_csv_path,arr,delimiter='-',comments='')

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