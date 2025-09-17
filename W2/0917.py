import os 
import numpy as np
import pandas as pd
from typing  import List
from sklearn.datasets import load_iris

ROOT= "iris_course"
ARTIFACTS = os.path.join(ROOT,'artifacts')
os.makedirs(ARTIFACTS,exist_ok=True)

def describe_stats(X:np.ndarray,names:List[str],title:str):
    m,s = X.mean(axis=0), X.std(axis=0)
    print(f"/n[{title}]")
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
    def __init__(self, in_dim=4, hidden1=64,hidden2=32,out_dim=3,dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,hidden1),nn.ReLU(),nn.Dropout(dropout),
            nn.Linear(hidden1,hidden2),nn.ReLU(),nn.Dropout(dropout),
            nn.Linear(hidden2,out_dim)
        )
    
    def forward(self,x): return self.net(x)

def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IrisMLP().to(device)
print(model)
print(f"可訓練參數量:{count_trainable_params(model):,}")

arch_txt =os.path.join(MODELS,"model_arch.txt")
with open(arch_txt,"w",encoding="utf-8") as f:
    f.write(str(model) + "\n")
    f.write(f"trainable_params={count_trainable_params(model)}\n")
print(f"-> 已存結構描述:{arch_txt}")
print("STEP5 完成")


       
