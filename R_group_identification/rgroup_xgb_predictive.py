import pandas as pd
import numpy as np
import re
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix
)

# === 1. Đọc dữ liệu CSV ===
df = pd.read_csv("rgroup_labeled_enrichment.csv")
df = df.dropna(subset=["rgroup_smiles", "Label"])
df["Label"] = df["Label"].astype(int)
print(f"👉 Tổng số R-group: {len(df)}")

# === 2. Tiền xử lý SMILES ===
def strip_attachment(smi: str) -> str:
    return re.sub(r"\[\*:[0-9]+\]", "", smi)

def smiles_to_morgan(smi, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros((nBits,))
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

df["clean_smi"] = df["rgroup_smiles"].apply(strip_attachment)
X = np.array([smiles_to_morgan(smi) for smi in df["clean_smi"]])
y = df["Label"].values

# === 3. Hàm train & evaluate XGBoost ===
def run_once(random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_prob)
    auprc = average_precision_score(y_test, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "Accuracy": acc,
        "MCC": mcc,
        "AUROC": auroc,
        "AUPRC": auprc,
        "Sensitivity": sensitivity,
        "Specificity": specificity
    }

# === 4. Chạy 3 lần ===
all_metrics = []
for seed in [42, 43, 44]:
    all_metrics.append(run_once(seed))

# === 5. Tính mean ± SD ===
df_metrics = pd.DataFrame(all_metrics)
mean_metrics = df_metrics.mean()
std_metrics = df_metrics.std()

summary = {metric: [f"{mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}"]
           for metric in mean_metrics.index}
summary_df = pd.DataFrame(summary)

# Lưu chỉ 1 file: mean ± SD
summary_df.to_csv("rgroup_labeled_enrichment_xgb_metrics.csv", index=False)

print("\n✅ Đã lưu file: xgb_rgroup_test_metrics_mean_sd.csv")
print("\n📊 Kết quả mean ± SD:")
print(summary_df)
