import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from rnn import SimpleRNN  # 匯入你的 rnn.py 模型

print("=" * 60)
print("✈️ RNN訓練範例 — 使用 AirPassengers 時間序列資料")
print("=" * 60)

# === 超參數設定 ===
SEQUENCE_LENGTH = 12   # 使用前12個月預測下一個月
INPUT_SIZE = 1
OUTPUT_SIZE = 1
HIDDEN_SIZE = 64
NUM_LAYERS = 1
NUM_EPOCHS = 200
LEARNING_RATE = 0.001
BATCH_SIZE = 16

print(f"\n超參數設定:")
print(f"序列長度: {SEQUENCE_LENGTH}")
print(f"隱藏層大小: {HIDDEN_SIZE}")
print(f"訓練輪數: {NUM_EPOCHS}")
print(f"學習率: {LEARNING_RATE}")
print(f"批次大小: {BATCH_SIZE}")

print("\n" + "-" * 60)
print("🔹 讀取資料集 data/AirPassengers.csv")
print("-" * 60)

# === 1️⃣ 讀取資料 ===
data_path = "./data/AirPassengers.csv"
df = pd.read_csv(data_path)

# 資料結構：Month,Passengers
if "Passengers" in df.columns:
    values = df["Passengers"].values.astype(float)
else:
    values = df.iloc[:, 1].values.astype(float)

# === 2️⃣ 正規化 ===
scaler = MinMaxScaler(feature_range=(0, 1))
values_scaled = scaler.fit_transform(values.reshape(-1, 1))

# === 3️⃣ 建立序列資料 ===
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(values_scaled, SEQUENCE_LENGTH)
X = torch.FloatTensor(X)
y = torch.FloatTensor(y)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"✅ 訓練資料大小: {X_train.shape}, 測試資料大小: {X_test.shape}")

# === DataLoader ===
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"每個 epoch 約 {len(train_loader)} 個批次")

# === 4️⃣ 建立模型 ===
model = SimpleRNN(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    output_size=OUTPUT_SIZE
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\n" + "-" * 60)
print("🔹 開始訓練")
print("-" * 60)

loss_history = []

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        output, _ = model(batch_X)
        prediction = output[:, -1, :]
        loss = criterion(prediction, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    loss_history.append(avg_loss)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Loss: {avg_loss:.6f}")

print("\n✅ 訓練完成！")

# === 儲存模型 ===
os.makedirs("./model", exist_ok=True)
model_path = "./model/rnn_air_model.pth"
torch.save(model.state_dict(), model_path)
print(f"模型已儲存至: {model_path}")


# === 繪製損失曲線 ===
os.makedirs("./result", exist_ok=True)  # ← 加這行！
plt.figure(figsize=(10, 5))
plt.plot(loss_history, color="blue")
plt.title("Training Loss Over Time", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss (MSE)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./result/training_loss_air.png", dpi=150)
print("📈 損失曲線已儲存至: ./result/training_loss_air.png")

# === 預測與反正規化 ===
model.eval()
with torch.no_grad():
    pred, _ = model(X_test)
    pred_y = pred[:, -1, :].numpy()
    pred_y = scaler.inverse_transform(pred_y)
    true_y = scaler.inverse_transform(y_test.numpy())

# === 繪製預測結果 ===
plt.figure(figsize=(12, 5))
plt.plot(true_y, 'g-', label='True', linewidth=2)
plt.plot(pred_y, 'r--', label='Predicted', linewidth=2)
plt.title("RNN Prediction on AirPassengers", fontsize=14)
plt.xlabel("Time Step", fontsize=12)
plt.ylabel("Passengers", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./result/air_rnn_prediction.png", dpi=150)
print("📊 預測結果已儲存至: ./result/air_rnn_prediction.png")

print("\n" + "=" * 60)
print("🎯 訓練流程完成！")
print("📁 生成的檔案：")
print("  ✅ rnn_air_model.pth         → 訓練後的模型")
print("  ✅ training_loss_air.png      → 損失曲線圖")
print("  ✅ air_rnn_prediction.png     → 預測結果圖")
print("=" * 60)
