import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)
n = 100
x = torch.randn(n,1)
# print('x',x)

# print(torch.tensor(10.0))
# print(torch.tensor([10.0]))

w = torch.tensor([10.0])
b = torch.tensor([3.0])

y = w * x + b + torch.randn(n,1) * 2.5

plt.figure()
# tensor 轉成 python 可以 的 numpy
plt.plot(x.numpy(),y.numpy(),'o',color='r')
plt.title('Data(x,y)')
plt.xlabel('x')
plt.ylabel('y')
# 自動排版器 圖表美觀
plt.tight_layout()
plt.savefig("step1_scatter.png")
plt.show()