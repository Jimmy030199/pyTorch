import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

# === 設定資料路徑與類別名稱 ===
data_root = './datasets2/cifar10_folders'
train_dir = os.path.join(data_root, 'train')
test_dir = os.path.join(data_root, 'test')

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


# === 自動建立 CIFAR-10 資料夾與圖片 ===
def generate_cifar10_folders():
    print("🔄 建立 CIFAR-10 資料夾與圖片中...")
    transform_to_pil = transforms.ToPILImage()

    cifar_train = torchvision.datasets.CIFAR10(
        root='./datasets2/cifar10',
        train=True,
        download=True
    )
    cifar_test = torchvision.datasets.CIFAR10(
        root='./datasets2/cifar10',
        train=False,
        download=True
    )

    for class_name in classes:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    # === 儲存訓練資料（每類 500 張） ===
    print("🖼️ 產生 train 圖片中...")
    class_counts_train = {i: 0 for i in range(10)}
    for idx, (img, label) in enumerate(cifar_train):
        if class_counts_train[label] < 500:
            if isinstance(img, torch.Tensor):
                img = transform_to_pil(img)
            img.save(os.path.join(train_dir, classes[label], f'{class_counts_train[label]}.png'))
            class_counts_train[label] += 1
        if all(c >= 500 for c in class_counts_train.values()):
            break

    # === 儲存測試資料（每類 100 張） ===
    print("🖼️ 產生 test 圖片中...")
    class_counts_test = {i: 0 for i in range(10)}
    for idx, (img, label) in enumerate(cifar_test):
        if class_counts_test[label] < 100:
            if isinstance(img, torch.Tensor):
                img = transform_to_pil(img)
            img.save(os.path.join(test_dir, classes[label], f'{class_counts_test[label]}.png'))
            class_counts_test[label] += 1
        if all(c >= 100 for c in class_counts_test.values()):
            break

    print("✅ CIFAR-10 圖片建立完成！")


# === 檢查資料夾是否含有圖片 ===
def has_images(folder):
    if not os.path.exists(folder):
        return False
    return any(
        fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        for _, _, files in os.walk(folder)
        for fname in files
    )


# === 主流程：檢查 train/test 是否有圖，否則自動生成 ===
if not has_images(train_dir) or not has_images(test_dir):
    generate_cifar10_folders()
else:
    print("✅ CIFAR-10 train/test 資料夾已存在且包含圖片。")


# === 資料前處理 ===
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

print("train: Resize + RandomFlip + RandomRotation + ToTensor + Normalize")
print("test: Resize + ToTensor + Normalize")


# === 載入 ImageFolder 資料 ===
full_train_dataset = ImageFolder(
    root=train_dir,
    transform=train_transform
)

test_dataset = ImageFolder(
    root=test_dir,
    transform=test_transform
)

print(f"full train size: {len(full_train_dataset)}")
print(f"test size: {len(test_dataset)}")
print(f"class_to_idx: {full_train_dataset.class_to_idx}")
print(f"classes: {full_train_dataset.classes}")


# === 切分 train / val (85% / 15%) ===
train_size = int(0.85 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"train dataset size = {len(train_dataset)}")
print(f"validation dataset size = {len(val_dataset)}")
print(f"test dataset size = {len(test_dataset)}")


# === DataLoader ===
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"train batches = {len(train_loader)}")
print(f"validation batches = {len(val_loader)}")
print(f"test batches = {len(test_loader)}")


# === 顯示訓練圖片樣本 ===
def imshow(img):
    img = img / 2 + 0.5  # 反正規化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')


dataiter = iter(train_loader)
images, labels = next(dataiter)

fig = plt.figure(figsize=(12, 6))
fig.suptitle('ImageFolder example', fontsize=16, fontweight='bold')

for idx in range(8):
    ax = plt.subplot(2, 4, idx + 1)
    img = images[idx]
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(f'class: {classes[labels[idx]]}')
    plt.axis('off')

plt.tight_layout()
os.makedirs('./outputs2', exist_ok=True)
plt.savefig('./outputs2/imagefolder_samples.png', dpi=150, bbox_inches='tight')
print("✅ saved: ./outputs2/imagefolder_samples.png")
