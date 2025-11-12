import os
import torch
import torchvision
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# === 資料路徑 ===
data_root = './datasets3'
images_dir = os.path.join(data_root, 'images')
csv_path = os.path.join(data_root, 'cifar10_labels.csv')

# === 類別名稱 ===
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# === 若尚未建立 CSV，從 CIFAR-10 建立資料集 ===
if not os.path.exists(csv_path):
    print("🔄 建立新的 CIFAR-10 資料與 CSV...")
    os.makedirs(images_dir, exist_ok=True)

    cifar_dataset = torchvision.datasets.CIFAR10(
        root=os.path.join(data_root, 'cifar10'),
        train=True,
        download=True
    )

    data_list = []
    class_counts = {i: 0 for i in range(10)}

    # 每類取前 10 張圖片作為示範
    for idx, (img, label) in enumerate(cifar_dataset):
        if class_counts[label] < 10:
            img_name = f"{classes[label]}_{class_counts[label]}.png"
            img_path = os.path.join(images_dir, img_name)
            img.save(img_path)

            data_list.append({
                'image_name': img_name,
                'label': label,
                'class_name': classes[label]
            })
            class_counts[label] += 1

        if all(count >= 10 for count in class_counts.values()):
            break

    df = pd.DataFrame(data_list)
    df.to_csv(csv_path, index=False)
    print(f"✅ Created {len(df)} data entries")
    print(f"📄 CSV file: {csv_path}")
    print(f"🖼️ Image directory: {images_dir}")
else:
    print("✅ Data already exists, skipping creation step.")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} entries from {csv_path}")


# === 自訂 Dataset 類別 ===
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        print(f"📦 Loaded {len(self.data_frame)} samples")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = self.data_frame.iloc[idx]['image_name']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = int(self.data_frame.iloc[idx]['label'])

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_name(self, idx):
        return self.data_frame.iloc[idx]['class_name']


# === 資料轉換與 DataLoader ===
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

dataset = CustomImageDataset(csv_file=csv_path, img_dir=images_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

print(f"✅ Dataset ready with {len(dataset)} samples")


# === 顯示 8 張圖片 ===
def imshow(img):
    img = img / 2 + 0.5  # 反正規化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')

# 取得一批資料
dataiter = iter(dataloader)
images, labels = next(dataiter)

fig = plt.figure(figsize=(12, 6))
fig.suptitle('Custom Dataset Example', fontsize=16, fontweight='bold')

for idx in range(8):
    ax = plt.subplot(2, 4, idx + 1)
    img = images[idx] / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(f'class: {classes[labels[idx]]}')
    plt.axis('off')

plt.tight_layout()

# 🔹 儲存到 outputs3
os.makedirs('./outputs3', exist_ok=True)
plt.savefig('./outputs3/customdataset_samples.png', dpi=150, bbox_inches='tight')
print("✅ saved: ./outputs3/customdataset_samples.png")

# 若你想同時在螢幕顯示：
plt.show()
