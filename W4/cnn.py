import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # 28x28 -> 14x14 -> 7x7
        # 輸入圖片[batch, 1, 28, 28] -> 第一層卷積[batch, 32, 28, 28]->第二層卷積[batch, 32, 28, 28]->
        # MaxPool2d[batch, 32, 14, 14]
        # 第三層卷積[batch, 64, 14, 14] -> 第四層卷積[batch, 64, 14, 14]
        # MaxPool2d[batch, 64, 7, 7]
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 輸入: 1 通道 (灰階) -> 32 feature maps
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 再接一層 conv
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 28 -> 14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32 -> 64 feature maps
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 14 -> 7

            nn.Dropout(0.25)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                           # 展平成一維向量
            nn.Linear(64 * 7 * 7, 256),             # 64*7*7 = 3136 -> 256
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)             # 輸出 10 類別 (Fashion-MNIST)
        )

    def forward(self, x):
        x = self.features(x)      # 卷積特徵抽取
        x = self.classifier(x)    # 全連接分類
        return x

# 通常會搭配：
# if __name__ == "__main__":
#     # 只有在「直接執行」這個檔案時才會跑
#     print("我是主程式")


# 這樣設計的好處：
# 如果你只是 import 這個檔案當作模組 → 不會執行裡面的測試程式。
# 如果你直接跑它 → 會執行 if 裡的測試程式。    
if __name__ == "__main__":
    # 快速檢測
    m = CNN()
    x = torch.randn(8,1,28,28) #產生一個 4 維張量 (tensor)
    y =m(x)
    print(y.shape)