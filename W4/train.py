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
from torch.utils.data import random_split, DataLoader


def main():
    # ===== Step1: 讓命令列 可以輸入參數 =====
    # 這段程式碼的功能是讓你的程式可以從命令列接收 資料夾路徑、訓練設定（epochs, batch_size, lr） 等參數，
    # 並自動建立需要的資料夾。這樣可以避免寫死參數，讓程式更靈活。

    #  這行使用 argparse 套件，建立一個解析器物件，可以讓你的程式支援 命令列參數。
    # 👉 例如你執行程式時可以這樣輸入：python train.py --epochs 10 --lr 0.01
    parser = argparse.ArgumentParser()


    #這裡定義了 5 個參數：
    # --data_dir：字串型別，預設值 "data"，用來指定存放或讀取資料集的目錄。
    # --out_dir：字串型別，預設值 "result"，用來指定輸出模型或訓練結果的目錄。
    # --epochs：整數型別，預設值 5，表示訓練迴圈要跑幾次。
    # --batch_size：整數型別，預設值 128，一個訓練批次的資料量。
    # --lr：浮點數型別，預設值 1e-3 (也就是 0.001)，學習率。
    # 📌 help 參數的文字會在使用 python train.py --help 時顯示，方便使用者理解
    parser.add_argument('--data_dir', type=str, default='data', help='資料下載/讀取目錄')
    parser.add_argument('--out_dir', type=str, default='result', help='訓練輸出目錄')
    parser.add_argument('--epochs', type=int, default=10,help='訓練回合數')
    parser.add_argument('--batch_size', type=int, default=128,help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3,help='學習率')

    # 這行會讀取你在命令列輸入的參數，並存進 args 物件。
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"data_dir={args.data_dir}, out_dir={args.out_dir}")



    # ===== Step2: 載入資料集 =====
    # torchvision.transforms
    # 這個模組提供了資料轉換 (preprocessing/augmentation) 的工具，通常會在讀入圖片時同時做處理。

    # 定義資料前處理 (transforms)

    # transforms.ToTensor()
    # 把 PIL 圖片或 numpy array 轉成 PyTorch Tensor。
    # 同時會把像素值從 0~255 縮放到 0~1 的範圍。
    # 例如：原始像素 128 → 128/255 ≈ 0.502。

    # transforms.Normalize((0.5,), (0.5,))
    # 這樣做的目的是：讓資料 平均值靠近 0、分布大致在 [-1,1]，有助於模型更快收斂。
    # RGB 彩色圖片 (3 個通道)：就要提供三個值。
    # 例如最常見的設定：
    # transforms.Normalize((0.5, 0.5, 0.5),   # 每個通道的 mean
    #                      (0.5, 0.5, 0.5))   # 每個通道的 std
    # 這樣 R/G/B 三個通道都會做 (x - 0.5)/0.5 → [-1, 1]。
    tfm = transforms.Compose([
        transforms.ToTensor(),                       # 將圖片轉成 tensor (0~1 浮點數)
        transforms.Normalize((0.5,), (0.5,))         # 灰階單通道標準化 (平均0.5, 標準差0.5)
    ])

    # # 載入 Fashion-MNIST 訓練資料集
    # train_dataset = datasets.FashionMNIST(
    #     root=args.data_dir,       # 資料存放位置 (前面 Step1 定義的參數)
    #     train=True,               # 載入訓練集 (True=訓練, False=測試)
    #     download=False,           # 若沒有資料就自動下載
    #     transform=tfm             # 套用前處理 (tensor + normalize)
    # )

    # # 印出訓練資料集大小 & 存放位置
    # print(f"訓練集筆數: {len(train_dataset)} (下載位置: {args.data_dir})")

    # 載入 Fashion-MNIST 訓練集
    full_train = datasets.FashionMNIST(
        root=args.data_dir,
        train=True,
        download=True,
        transform=tfm
    )

    # 分割資料集為訓練與驗證
    val_len = 10000
    train_len = len(full_train) - val_len
    train_set, val_set = random_split(full_train, [train_len, val_len])

    # 指定運算裝置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ===== Step3: 建立 DataLoader =====

    # num_workers=2
    # 用多少個子程序平行載入資料。
    # 數字越大 → 資料讀取越快，但也吃更多 CPU。
    # 在 Windows 上有時要小心，num_workers>0 需要 if __name__ == "__main__": 包起來，不然可能會卡住

    # pin_memory=True
    # 當你在 GPU 上訓練時，會把 batch 固定在 page-locked memory，加速 CPU → GPU 的資料傳輸。
    # 在 CUDA 環境下通常建議開啟。

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,       # 每個 epoch 打亂資料
    #     num_workers=2,      # 使用兩個子程序加速讀取
    #     pin_memory=True     # 在 GPU 上訓練時建議開啟
    # )
    # print(f"DataLoader 就緒 (batch_size={args.batch_size})")

    # 建立 DataLoader
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=True
    )

    val_loader = DataLoader(
        val_set, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )

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
    best_val_loss=float('inf')
    best_path=os.path.join(args.out_dir,'best_cnn.pth')
    
    train_acc = 0.0
    # 開始訓練
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()    # 切換到訓練模式 (啟用 dropout, BN)
        running_loss, n = 0.0, 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).long()   # CrossEntropyLoss 需要 int64

            # Forward
            logits = model(images)              # 模型輸出 [batch, 10] 輸出的「原始分數」
            loss = criterion(logits, labels)    # 計算損失 (未 softmax)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累積損失 (乘 batch size，換算成總和)

            # labels.size(0) 取得這個 batch 的大小（batch size）。
            # 假設 labels.shape = [128]，那 bs = 128。
            # 👉 這樣不管最後 batch 有幾筆，都能動態取得數量。
            bs = labels.size(0)

            
            # 假設 batch loss = 0.5678
            # loss：tensor(0.5678, grad_fn=<NllLossBackward0>)（可以做 backward）
            # loss.item()：0.5678（純數字，可以加總統計）
            running_loss += loss.item() * bs
            n += bs
        
        
        
        # 計算平均訓練損失
        train_loss = running_loss / n
        train_acc=train_acc / n
        # --- validate ---
        model.eval()
        val_loss, val_acc = 0.0, 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)

                # 計算 batch 的 loss 與正確率
                val_loss += loss.item()
                val_acc += (logits.argmax(1) == labels).float().sum().item()

        # 平均 loss、acc
        val_loss /= len(val_loader)
        val_acc /= len(val_loader.dataset)

        # 將結果寫入 csv 檔
        with open(loss_csv, 'a', encoding='utf-8') as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{train_acc:.4f},{val_acc:.4f}\n")

        # 顯示目前 epoch 的結果
        print(f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        # 若 validation loss 下降，則儲存最好的模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, best_path)
            print(f"--> Saved new best to {best_path}")


        # 記錄到 CSV
        with open(loss_csv, 'a', encoding='utf-8') as f:
            f.write(f'{epoch},{train_loss:.6f}\n')

        # 印出每個 epoch 結果
        dt = time.time() - t0
        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f} ({dt:.1f}s)")

    print(f"✅ 訓練完成，loss 已寫入: {loss_csv}")


if __name__ == '__main__':
    main()






