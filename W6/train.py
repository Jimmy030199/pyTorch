import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

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


