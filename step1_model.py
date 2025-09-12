import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, in_dim=1,out_dim=1):
        # 呼叫父類別 建構子
        super().__init__()
        self.lin = nn.Linear(in_dim,out_dim)
        #nn.Module 和 nn.Linear 不是同一層級，
        #nn.Linear 繼承自 nn.Module。
        
    # x 是模型的輸入資料 pytorch 會自動把資料丟到forward
    def forward(self,x):
        return self.lin(x)
model = Model(1,1)
# print(model)

x =torch.tensor([[1.0],[2.0],[3.0]])
y = model(x)
# print(y)



