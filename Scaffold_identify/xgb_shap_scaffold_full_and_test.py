import os
import random
import numpy as np
import pandas as pd
import shap
from datetime import datetime
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem, DataStructs
import xgboost as xgb
import time

# ==== 1. Set seed for reproducibility ====
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

# ==== 2. Scaffold + ECFP generation ====
def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol)) if mol else None

def smiles_to_ecfp(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    return np.zeros(n_bits)

# ==== 3. Scaffold split ====
def scaffold_split(df, test_size=0.2, seed=42):
    """Split theo scaffold, để test scaffolds không trùng train"""
    scaffolds = df['scaffold'].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(scaffolds)

    n_test = int(len(scaffolds) * test_size)
    test_scaffolds = scaffolds[:n_test]
    train_scaffolds = scaffolds[n_test:]

    df_train = df[df['scaffold'].isin(train_scaffolds)].copy()
    df_test = df[df['scaffold'].isin(test_scaffolds)].copy()

    return df_train, df_test

# ==== 4. Main SHAP analysis pipeline with XGB + scaffold split ====
def main(test_size=0.2, random_state=42):
    set_seed(random_state)

    # === Load and process data ===
    df = pd.read_csv("/home/andy/andy/Inflam_NP/Scaffold_identify/InFlam_full.csv").dropna(subset=['canonical_smiles', 'Label'])
    df['scaffold'] = df['canonical_smiles'].apply(get_scaffold)
    df = df.dropna(subset=['scaffold'])

    df['fingerprint'] = df['canonical_smiles'].apply(smiles_to_ecfp)

    # === Scaffold split ===
    df_train, df_test = scaffold_split(df, test_size=test_size, seed=random_state)

    X_train = np.array(df_train['fingerprint'].tolist())
    y_train = df_train['Label'].astype(int).values

    X_test = np.array(df_test['fingerprint'].tolist())
    y_test = df_test['Label'].astype(int).values

    X_full = np.array(df['fingerprint'].tolist())
    y_full = df['Label'].astype(int).values

    # === Train XGBoost ===
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

    explainer = shap.TreeExplainer(model)

    # === Output folder ===
    output_dir = f"shap_scaffold_split_XGB_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # === 1. SHAP on full dataset ===
    shap_values_full = explainer.shap_values(X_full)
    per_sample_mean_full = shap_values_full.mean(axis=1)
    df_shap_full = pd.DataFrame({
        "scaffold": df['scaffold'].values,
        "mean_shap": per_sample_mean_full
    })
    df_summary_full = (df_shap_full.groupby("scaffold")["mean_shap"]
                       .mean().reset_index()
                       .sort_values(by="mean_shap", ascending=False))
    df_summary_full["effect"] = df_summary_full["mean_shap"].apply(lambda x: "positive" if x > 0 else "negative")

    df_shap_full.to_csv(f"{output_dir}/scaffold_shap_per_sample_full.csv", index=False)
    df_summary_full.to_csv(f"{output_dir}/scaffold_shap_summary_full.csv", index=False)

    # === 2. SHAP on test set (scaffold unseen) ===
    shap_values_test = explainer.shap_values(X_test)
    per_sample_mean_test = shap_values_test.mean(axis=1)
    df_shap_test = pd.DataFrame({
        "scaffold": df_test['scaffold'].values,
        "mean_shap": per_sample_mean_test
    })
    df_summary_test = (df_shap_test.groupby("scaffold")["mean_shap"]
                       .mean().reset_index()
                       .sort_values(by="mean_shap", ascending=False))
    df_summary_test["effect"] = df_summary_test["mean_shap"].apply(lambda x: "positive" if x > 0 else "negative")

    df_shap_test.to_csv(f"{output_dir}/scaffold_shap_per_sample_test.csv", index=False)
    df_summary_test.to_csv(f"{output_dir}/scaffold_shap_summary_test.csv", index=False)

    # === Log info ===
    print(f"\n✅ SHAP done with scaffold split. Results saved in: {output_dir}")
    print(f"📊 Full dataset: {len(df)} molecules, {df['scaffold'].nunique()} scaffolds")
    print(f"📊 Train set: {len(df_train)} molecules, {df_train['scaffold'].nunique()} scaffolds")
    print(f"📊 Test set: {len(df_test)} molecules, {df_test['scaffold'].nunique()} scaffolds")
    print(f"💾 Full summary   → scaffold_shap_summary_full.csv")
    print(f"💾 Test summary   → scaffold_shap_summary_test.csv")

if __name__ == "__main__":
    main(test_size=0.2, random_state=42)
