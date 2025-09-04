import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, matthews_corrcoef,
    roc_auc_score, precision_score, precision_recall_curve,
    balanced_accuracy_score, auc
)

# === 1. Load data ===
df_train = pd.read_csv("InFlam_full_x_train.csv")
df_test = pd.read_csv("InFlam_full_x_test.csv")

# === 2. Scaffold & fingerprint functions ===
def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
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

# === 6. Lặp huấn luyện với LGBM ===
NUM_RUNS = 3
results = []

for run in range(NUM_RUNS):
    seed = 42 + run
    model = LGBMClassifier(class_weight='balanced', random_state=seed, n_estimators=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan

    # AUPRC
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(rec_arr, prec_arr)

    # Metrics
    result = {
        "Run": run + 1,
        "Seed": seed,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
        "AUROC": roc_auc_score(y_test, y_prob),
        "AUPRC": pr_auc,
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Sensitivity": sensitivity,
        "Specificity": specificity
    }

    results.append(result)

# === 7. Lưu kết quả từng lần chạy
df_raw = pd.DataFrame(results)
df_raw.to_csv("lgbm_metrics_ecfp+scaffold_raw.csv", index=False)
print("✅ Đã lưu kết quả từng lần chạy → lgbm_metrics_ecfp+scaffold_raw.csv")

# === 8. Tính trung bình ± SD và lưu theo định dạng 1 dòng
metrics_order = [
    "Accuracy", "Balanced Accuracy", "AUROC", "AUPRC",
    "MCC", "Precision", "Sensitivity", "Specificity"
]

df_summary_stats = df_raw[metrics_order].agg(['mean', 'std']).T
df_summary_stats["Scaffold"] = df_summary_stats["mean"].round(3).astype(str) + " ± " + df_summary_stats["std"].round(3).astype(str)

# Dạng hàng là "Scaffold", cột là các metric
df_summary = df_summary_stats[["Scaffold"]].T
df_summary.index = ["Scaffold"]

df_summary.to_csv("lgbm_metrics_ecfp+scaffold_summary.csv")
print("✅ Đã lưu summary → lgbm_metrics_ecfp+scaffold_summary.csv")

# === 9. In ra terminal
print("\n📊 Trung bình ± SD trên", NUM_RUNS, "lần chạy:")
print(df_summary)
