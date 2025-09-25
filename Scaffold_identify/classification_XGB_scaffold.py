import pandas as pd 
import numpy as np
import os
from datetime import datetime
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, matthews_corrcoef,
    roc_auc_score, precision_score, precision_recall_curve,
    balanced_accuracy_score, auc
)
from xgboost import XGBClassifier
from typing import Optional

# === 0. Output folder ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"xgb_results_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# === 1. Load data ===
df_train = pd.read_csv("InFlam_full_x_train.csv")
df_test = pd.read_csv("InFlam_full_x_test.csv")

# === 2. Scaffold & fingerprint functions ===
def _standardize_mol(mol: Chem.Mol) -> Optional[Chem.Mol]:
    if mol is None:
        return None
    try:
        params = rdMolStandardize.CleanupParameters()
        mol = rdMolStandardize.Cleanup(mol, params)
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
        return mol
    except Exception:
        return None

def get_scaffold(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    mol = _standardize_mol(mol)
    if mol is None:
        return None
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        if core is None or core.GetNumAtoms() == 0:
            return None
        return Chem.MolToSmiles(core, isomericSmiles=False,
                                kekuleSmiles=False, canonical=True)
    except Exception:
        return None

def mol_to_ecfp(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    mol = _standardize_mol(mol)
    if not mol:
        return np.zeros((nBits,))
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    arr = np.zeros((nBits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

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

# === 5. Tạo hai bộ dữ liệu: combine và scaffold-only ===
X_train_combined = np.hstack([X_ecfp_train, X_scaffold_train])
X_test_combined = np.hstack([X_ecfp_test, X_scaffold_test])

X_train_scaffold = X_scaffold_train
X_test_scaffold = X_scaffold_test

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

# === 7. Hàm chạy nhiều seed và tính metrics ===
def run_experiment(X_train, y_train, X_test, y_test, name, output_dir):
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

    # Lưu raw
    df_raw = pd.DataFrame(results)
    df_raw.to_csv(os.path.join(output_dir, f"xgb_metrics_{name}_raw.csv"), index=False)

    # Tính mean ± std
    metrics_order = [
        "Accuracy", "Balanced Accuracy", "AUROC", "AUPRC",
        "MCC", "Precision", "Sensitivity", "Specificity"
    ]
    df_summary_stats = df_raw[metrics_order].agg(['mean', 'std']).T
    df_summary_stats[name] = (
        df_summary_stats["mean"].round(3).astype(str) + " ± " + df_summary_stats["std"].round(3).astype(str)
    )
    df_summary = df_summary_stats[[name]].T
    df_summary.index = [name]
    df_summary.to_csv(os.path.join(output_dir, f"xgb_metrics_{name}_summary.csv"))
    print(f"✅ Đã lưu summary cho {name} → {output_dir}/xgb_metrics_{name}_summary.csv")
    return df_summary

# === 8. Chạy 2 mô hình ===
summary_combined = run_experiment(X_train_combined, y_train, X_test_combined, y_test, "ecfp+scaffold", output_dir)
summary_scaffold = run_experiment(X_train_scaffold, y_train, X_test_scaffold, y_test, "scaffold_only", output_dir)

# === 9. Gộp kết quả ===
df_summary_all = pd.concat([summary_combined, summary_scaffold])
df_summary_all.to_csv(os.path.join(output_dir, "xgb_metrics_all_summary.csv"))
print(f"\n📊 Trung bình ± SD trên 3 lần chạy (XGBoost):")
print(df_summary_all)
print(f"\n📁 Tất cả kết quả được lưu tại: {output_dir}")
