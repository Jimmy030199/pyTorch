import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
import numpy as np

# === 資料轉換設定 ===
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# === 載入 CIFAR-10 訓練集與測試集 ===
trainset = torchvision.datasets.CIFAR10(
    root='./datasets/cifar10',
    train=True,
    download=True,
    transform=transform
)

testset = torchvision.datasets.CIFAR10(
    root='./datasets/cifar10',
    train=False,
    download=True,
    transform=transform
)

# === 顯示資料集基本資訊 ===
print(f"train data size = {len(trainset)}")
print(f"test data size = {len(testset)}")
print(f"class count = {len(trainset.classes)}")
print("class names =", trainset.classes)
print("class_to_idx =", trainset.class_to_idx)

# === 取出一筆樣本 ===
img, lbl = trainset[0]
print(f"single image shape = {img.shape}, label = {lbl}, class name = {trainset.classes[lbl]}")


# === 切分訓練 / 驗證集 (80% / 20%) ===
train_size = int(0.8 * len(trainset))
val_size = len(trainset) - train_size

train_dataset, val_dataset = random_split(
    trainset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # 固定隨機種子確保可重現
)

print(f"train dataset size = {len(train_dataset)}")
print(f"validation dataset size = {len(val_dataset)}")
print(f"test dataset size = {len(testset)}")

# === 建立 DataLoader ===
batch_size = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

print(f"train batches = {len(train_loader)}")
print(f"validation batches = {len(val_loader)}")
print(f"test batches = {len(test_loader)}")

# === 顯示圖片的函式 ===
def imshow(img):
    img = img / 2 + 0.5  # 反正規化 (把[-1,1]轉回[0,1])
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')

# === 取出一批訓練圖片 ===
dataiter = iter(train_loader)
images, labels = next(dataiter)
classes = trainset.classes  # CIFAR-10 類別名稱

# === 繪製前8張圖片 ===
fig = plt.figure(figsize=(12, 6))
plt.suptitle('CIFAR-10 Samples', fontsize=16, fontweight='bold')

for idx in range(8):
    ax = plt.subplot(2, 4, idx + 1)
    img = images[idx] / 2 + 0.5  # 反正規化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(f'class: {classes[labels[idx]]}')
    plt.axis('off')

plt.tight_layout()

# === 儲存結果 ===
os.makedirs('./outputs', exist_ok=True)
plt.savefig('./outputs/cifar10_samples.png', dpi=150, bbox_inches='tight')
print("✅ saved:", './outputs/cifar10_samples.png')




