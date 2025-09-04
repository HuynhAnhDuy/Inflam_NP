import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, matthews_corrcoef,
    roc_auc_score, precision_score, precision_recall_curve,
    balanced_accuracy_score, auc
)
from xgboost import XGBClassifier

# === 1. Load data ===
df_train = pd.read_csv("InFlam_full_x_train.csv")
df_test = pd.read_csv("InFlam_full_x_test.csv")

# === 2. Scaffold & fingerprint functions ===
def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    return "None"

def mol_to_ecfp(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return np.zeros((nBits,))
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    return np.array(fp)

# === 3. Tính scaffold và ECFP ===
for df in [df_train, df_test]:
    df['scaffold'] = df['canonical_smiles'].apply(get_scaffold)
    df['ecfp'] = df['canonical_smiles'].apply(mol_to_ecfp)

X_ecfp_train = np.stack(df_train['ecfp'].values)
X_ecfp_test = np.stack(df_test['ecfp'].values)

# === 4. Encode scaffold ===
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_scaffold_train = ohe.fit_transform(df_train[['scaffold']])
X_scaffold_test = ohe.transform(df_test[['scaffold']])

# === 5. Combine ECFP + Scaffold ===
X_train = np.hstack([X_ecfp_train, X_scaffold_train])
X_test = np.hstack([X_ecfp_test, X_scaffold_test])
y_train = df_train['Label'].values
y_test = df_test['Label'].values

# === 6. Training function ===
def train_xgboost(X_train, y_train, X_test, seed):
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        gamma=0,
        min_child_weight=1,
        max_depth=6,
        objective="binary:logistic",
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    return y_pred, y_proba

# === 7. Run 3 seeds ===
results = []
for run, seed in enumerate([42, 43, 44], start=1):
    y_pred, y_proba = train_xgboost(X_train, y_train, X_test, seed)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    precision = precision_score(y_test, y_pred, zero_division=0)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)

    prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(rec_arr, prec_arr)

    results.append({
        "Run": run,
        "Seed": seed,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Balanced Accuracy": balanced_acc,
        "AUROC": roc_auc_score(y_test, y_proba),
        "AUPRC": pr_auc,
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Precision": precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity
    })

# === 8. Save raw results
df_raw = pd.DataFrame(results)
df_raw.to_csv("xgb_metrics_ecfp+scaffold_raw.csv", index=False)
print("✅ Đã lưu kết quả từng lần chạy → xgb_metrics_ecfp+scaffold_raw.csv")

# === 9. Tính trung bình ± SD và lưu
metrics_order = [
    "Accuracy", "Balanced Accuracy", "AUROC", "AUPRC",
    "MCC", "Precision", "Sensitivity", "Specificity"
]

df_summary_stats = df_raw[metrics_order].agg(['mean', 'std']).T
df_summary_stats["Scaffold"] = (
    df_summary_stats["mean"].round(3).astype(str) + " ± " + df_summary_stats["std"].round(3).astype(str)
)

# Đưa về format: hàng = Scaffold, cột = các metric
df_summary = df_summary_stats[["Scaffold"]].T
df_summary.index = ["Scaffold"]

df_summary.to_csv("xgb_metrics_ecfp+scaffold_summary.csv")
print("✅ Đã lưu summary → xgb_metrics_ecfp+scaffold_summary.csv")

# === 10. Print to terminal
print("\n📊 Trung bình ± SD trên 3 lần chạy (XGBoost):")
print(df_summary)
