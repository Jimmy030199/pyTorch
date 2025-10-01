import os
# 用來解析命令列參數，讓程式執行時可以用不同參數設定
import argparse 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from cnn import CNN   # 假設你把 CNN 類別存成 cnn.py
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='資料下載/讀取目錄')
    parser.add_argument('--out_dir', type=str, default='result', help='訓練輸出目錄')
    parser.add_argument('--epochs', type=int, default=5,help='訓練回合數')
    parser.add_argument('--batch_size', type=int, default=128,help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3,help='學習率')
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"data_dir={args.data_dir}, out_dir={args.out_dir}")
    # ===== Step2: 載入資料集 =====
    # 定義資料前處理 (transforms)
    tfm = transforms.Compose([
        transforms.ToTensor(),                       # 將圖片轉成 tensor (0~1 浮點數)
        transforms.Normalize((0.5,), (0.5,))         # 灰階單通道標準化 (平均0.5, 標準差0.5)
    ])

    # 載入 Fashion-MNIST 訓練資料集
    train_dataset = datasets.FashionMNIST(
        root=args.data_dir,       # 資料存放位置 (前面 Step1 定義的參數)
        train=True,               # 載入訓練集 (True=訓練, False=測試)
        download=False,            # 若沒有資料就自動下載
        transform=tfm             # 套用前處理 (tensor + normalize)
    )

    # 印出訓練資料集大小 & 存放位置
    print(f"訓練集筆數: {len(train_dataset)} (下載位置: {args.data_dir})")

    # ===== Step3: 建立 DataLoader =====
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,       # 每個 epoch 打亂資料
        num_workers=2,      # 使用兩個子程序加速讀取
        pin_memory=True     # 在 GPU 上訓練時建議開啟
    )
    print(f"DataLoader 就緒 (batch_size={args.batch_size})")

    # 選擇裝置 (GPU 優先，否則用 CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 建立模型 (Fashion-MNIST 有 10 類)
    model = CNN(num_classes=10).to(device)
    print(f"使用裝置: {device}")

    # 損失函數 (CrossEntropyLoss: 結合 softmax + NLLLoss)
    criterion = nn.CrossEntropyLoss()

    # 優化器 (Adam)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("[S5] Loss/Optimizer 準備完成")

    # 訓練損失輸出 CSV 檔案
    loss_csv = os.path.join(args.out_dir, 'loss.csv')
    with open(loss_csv, 'w', encoding='utf-8') as f:
        f.write('epoch,train_loss\n')

    # 開始訓練
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()    # 切換到訓練模式 (啟用 dropout, BN)
        running_loss, n = 0.0, 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).long()   # CrossEntropyLoss 需要 int64

            # Forward
            logits = model(images)              # 模型輸出 [batch, 10]
            loss = criterion(logits, labels)    # 計算損失 (未 softmax)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累積損失 (乘 batch size，換算成總和)
            bs = labels.size(0)
            running_loss += loss.item() * bs
            n += bs

        # 計算平均訓練損失
        train_loss = running_loss / n

        # 記錄到 CSV
        with open(loss_csv, 'a', encoding='utf-8') as f:
            f.write(f'{epoch},{train_loss:.6f}\n')

        # 印出每個 epoch 結果
        dt = time.time() - t0
        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f} ({dt:.1f}s)")

    print(f"✅ 訓練完成，loss 已寫入: {loss_csv}")


if __name__ == '__main__':
    main()






