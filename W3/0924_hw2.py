import os 
import numpy as np
import pandas as pd
from typing  import List
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

ROOT= "W3/iris_course"
ARTIFACTS = os.path.join(ROOT,'artifacts')
os.makedirs(ARTIFACTS,exist_ok=True)

# axis=0 與 axis=1 差別
# print("axis=0：每欄平均", X.mean(axis=0))
# print("axis=1：每列平均", X.mean(axis=1))

def describe_stats(X:np.ndarray,names:List[str],title:str):
    m,s = X.mean(axis=0), X.std(axis=0)
    print(f"\n[{title}]")
    for n, mi, sd in zip(names, m,s):
        print(f"{n:<14s} mean={mi:8.4f} std={sd:8.4f}" )

print("***Step1 載入資料與探索")
iris =load_iris()
x,y = iris.data,iris.target
feature_names = ["sepal_length","sepal_width","petal_length","petal_width"]
target_names=iris.target_names.tolist()

df=pd.DataFrame(x,columns=feature_names)
df["target"] = y
print("\n 前5筆資料");print(df.head())
print("\n 類別分布:")

for i ,name in enumerate(target_names):
    print(f"{i}={name:<10s} : {(y==i).sum()} 筆")
describe_stats(x,feature_names,"原始資料(未標準化)")

out_csv=os.path.join(ARTIFACTS,"iris_preview.csv")
df.head(20).to_csv(out_csv,index=False)
print(f"\n->已存取20筆預覽 :{out_csv}")
print("STEP1 完成")





from sklearn.model_selection import train_test_split
print("\n===STEP2 | 切分 Train/Val/Test===")
X_trainval,X_test,y_trainval,y_test =train_test_split(
    x,y,test_size=0.2,random_state=42,stratify=y
)
X_train,X_val,y_train,y_val =train_test_split(
    X_trainval,y_trainval,test_size=0.2,random_state=42,stratify=y_trainval
)

print(f"切分形狀: train={X_train.shape} val={X_val.shape} test={X_test.shape}")
print("STEP2 完成")




from sklearn.preprocessing import StandardScaler
import joblib

print('\====STEP 3 | 標準化(只用訓練集fit)並存檔)')
scaler = StandardScaler().fit(X_train)
X_train_sc = scaler.transform(X_train)
X_val_sc = scaler.transform(X_val)
X_test_sc = scaler.transform(X_test)

describe_stats(X_train, feature_names,"訓練集(標準化前)")
describe_stats(X_train_sc, feature_names,"訓練集(標準化後)")

npz_path= os.path.join(ARTIFACTS,"train_val_test_scaled.npz")
np.savez(npz_path,
         X_train_sc=X_train_sc,y_train=y_train,
         X_val_sc=X_val_sc,y_val=y_val,
         X_test_sc=X_test_sc,y_test=y_test,
         feature_names=np.array(feature_names,dtype=object),
         target_names=np.array(target_names,dtype=object),
)
scaler_path =os.path.join(ARTIFACTS,"scaler.pkl")
joblib.dump(scaler,scaler_path)

print(f"->已存標準化資料:{npz_path}")
print(f"->已存標準化器:{scaler_path}")
print("STEP3 完成")





import torch
from torch.utils.data import TensorDataset,DataLoader

print('\====STEP 4 | Tensor 與 DataLoader')
X_train_t = torch.tensor(X_train_sc,dtype=torch.float32)
y_train_t = torch.tensor(y_train,dtype=torch.long)
X_val_t = torch.tensor(X_val_sc,dtype=torch.float32)
y_val_t = torch.tensor(y_val,dtype=torch.long)

train_loader =DataLoader(TensorDataset(X_train_t,y_train_t),batch_size=16,shuffle=True)
val_loader =DataLoader(TensorDataset(X_val_t,y_val_t),batch_size=16,shuffle=False)

xb,yb = next(iter(train_loader))
print(f"第一個 batch:xb.shape={xb.shape}, yb.shape={yb.shape}")
print(f"   xb[0](標準化後)={xb[0].tolist()}")
print(f"   yb[0](類別)={yb[0].item()}")

batch_preview=os.path.join(ARTIFACTS,"batch_preview.csv")
pd.DataFrame(xb.numpy(),columns=feature_names).assign(label=yb.numpy()).to_csv(batch_preview,index=False)
print(f"->已存batch 預覽{batch_preview}")
print("STEP4 完成")


from torch import nn 
MODELS = os.path.join(ROOT,"models")
os.makedirs(MODELS,exist_ok=True)

print('\====STEP 5 | 定義模型與參數量')

class IrisMLP(nn.Module):
    def __init__(self, in_dim=4, hidden1=64,hidden2=32,out_dim=3,dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,hidden1),nn.ReLU(),nn.Dropout(dropout),
            nn.Linear(hidden1,hidden2),nn.ReLU(),nn.Dropout(dropout),
            nn.Linear(hidden2,out_dim)
        )
    
    def forward(self,x): return self.net(x)

def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model = IrisMLP().to(device)
print(model)
print(f"可訓練參數量:{count_trainable_params(model):,}")

arch_txt =os.path.join(MODELS,"model_arch.txt")
with open(arch_txt,"w",encoding="utf-8") as f:
    f.write(str(model) + "\n")
    f.write(f"trainable_params={count_trainable_params(model)}\n")
print(f"-> 已存結構描述:{arch_txt}")
print("STEP5 完成")


       
print("\n=== STEP 6 訓練 (含早停) ===")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def evaluate(m, loader):
    m.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = m(xb)
            print(logits)
            print(yb)
            loss_sum += criterion(logits, yb).item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
    return loss_sum / total, correct / total

best_state, best_val, patience, bad = None, -1.0, 15, 0
hist = {"tr_loss": [], "tr_acc": [], "va_loss": [], "va_acc": []}

for ep in range(1, 2001):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_sum += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)
    
    tr_loss, tr_acc = loss_sum / total, correct / total
    va_loss, va_acc = evaluate(model, val_loader)
    
    hist["tr_loss"].append(tr_loss)
    hist["tr_acc"].append(tr_acc)
    hist["va_loss"].append(va_loss)
    hist["va_acc"].append(va_acc)
    
    print(f"Epoch {ep:03d} | train_loss={tr_loss:.4f} acc={tr_acc:.3f} | val_loss={va_loss:.4f} acc={va_acc:.3f}")

    if va_acc > best_val:
     best_val, best_state,bad = va_acc, {k: v.cpu() for k, v in model.state_dict().items()},0
    else:
        bad += 1
        if bad >= patience:
            print(f"早停：{patience} epochs 未提升")
            break

# 載回最佳
if best_state is not None:
    model.load_state_dict(best_state)


# === 畫訓練/驗證曲線 ===
xs = np.arange(1, len(hist["tr_loss"]) + 1)
plt.figure(figsize=(8, 4))

# loss 曲線
plt.plot(xs, hist["tr_loss"], label="train_loss")
plt.plot(xs, hist["va_loss"], label="val_loss")

# acc 曲線
plt.plot(xs, hist["tr_acc"], label="train_acc")
plt.plot(xs, hist["va_acc"], label="val_acc")

plt.xlabel("epoch")
plt.ylabel("value")
plt.title("Training Curves")
plt.legend()
plt.tight_layout()

PLOTS = os.path.join(ROOT, "plots")
# 確保資料夾存在
os.makedirs(PLOTS, exist_ok=True)

# 儲存圖表
curve_path = os.path.join(PLOTS, "curves2.png")
plt.savefig(curve_path, dpi=150)
plt.close()
print(f"✅ 已存訓練曲線: {curve_path}")

# === 儲存最佳模型權重 ===
best_path = os.path.join(MODELS, "best.pt")
torch.save(model.state_dict(), best_path)
print(f"✅ 已存最佳權重: {best_path}")

print("STEP 6 ✅ 完成")

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("\n=== STEP 7 測試集評估 ===")
model.eval()
with torch.no_grad():
    logits = model(torch.tensor(X_test_sc, dtype=torch.float32).to(device))
    y_pred = logits.argmax(1).cpu().numpy()

# 計算準確率
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy = {acc:.3f}\n")

# 分類報告
print("分類報告：")
print(classification_report(y_test, y_pred, target_names=target_names, digits=3))

# === 混淆矩陣 ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4.5, 4))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.colorbar()

ticks = np.arange(len(target_names))
plt.xticks(ticks, target_names, rotation=30)
plt.yticks(ticks, target_names)

# 在矩陣格子中填上數字
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()

cm_path = os.path.join(PLOTS, "confusion_matrix2.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"->已存混淆矩陣: {cm_path}")

# === 回到原單位顯示幾筆 ===
X_test_orig = scaler.inverse_transform(X_test_sc)
show_n = min(10, len(X_test_orig))

tbl = pd.DataFrame(X_test_orig[:show_n], columns=feature_names)
tbl["True"]  = [target_names[i] for i in y_test[:show_n]]
tbl["Pred"]  = [target_names[i] for i in y_pred[:show_n]]

samples_csv = os.path.join(ARTIFACTS, "test_samples.csv")
tbl.to_csv(samples_csv, index=False)
print(f"已存樣本對照(原單位): {samples_csv}")

print("STEP 7 完成 ✅ （全流程）")

