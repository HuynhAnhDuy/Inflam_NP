import os
import random
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
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

# ==== 3. Main SHAP analysis pipeline with XGB ====
def main(mode="fast", num_runs=5, background_size=200):
    """
    mode="fast": 1 run, explain toàn bộ dataset
    mode khác: chạy num_runs lần và gom kết quả
    """
    set_seed()

    # === Load and process data ===
    df = pd.read_csv("3.InFlamNat_SHAP.csv").dropna(subset=['canonical_smiles', 'Label'])
    df['scaffold'] = df['canonical_smiles'].apply(get_scaffold)
    df = df.dropna(subset=['scaffold'])

    df['fingerprint'] = df['canonical_smiles'].apply(smiles_to_ecfp)
    X_array = np.array(df['fingerprint'].tolist())
    y_array = df['Label'].astype(int).values
    feature_names = [f'Bit_{i}' for i in range(X_array.shape[1])]
    X_df = pd.DataFrame(X_array, columns=feature_names)

    output_dir = f"shap_scaffold_analysis_XGB_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # ---- FAST MODE: single run ----
    if mode == "fast":
        run_index = 0
        print(f"\n🚀 FAST mode: single run on full dataset ({len(X_df)} molecules)")
        start_time = time.time()

        # Train XGBoost
        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss"
        )
        model.fit(X_df.values, y_array)

        # SHAP with TreeExplainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_df)

        # Map scaffold theo index
        explain_scaffold = df['scaffold'].values

        # Tính mean SHAP per-sample rồi gom theo scaffold
        per_sample_mean = shap_values.mean(axis=1)
        df_shap = pd.DataFrame({
            "scaffold": explain_scaffold,
            "mean_shap": per_sample_mean
        })
        df_summary = (df_shap.groupby("scaffold")["mean_shap"]
                      .mean().reset_index()
                      .sort_values(by="mean_shap", ascending=False))
        df_summary["effect"] = df_summary["mean_shap"].apply(lambda x: "positive" if x > 0 else "negative")

        # Save
        df_shap.to_csv(f"{output_dir}/scaffold_shap_per_sample.csv", index=False)
        df_summary.to_csv(f"{output_dir}/scaffold_shap_summary.csv", index=False)

        elapsed = time.time() - start_time
        print(f"✅ SHAP done for {len(X_df)} molecules in {elapsed:.2f} seconds")
        print(f"📌 Unique scaffolds analyzed: {df['scaffold'].nunique()}")
        print(f"💾 Saved per-sample → {output_dir}/scaffold_shap_per_sample.csv")
        print(f"💾 Saved summary   → {output_dir}/scaffold_shap_summary.csv")
        return

    # ---- Multi-run path ----
    all_scaffold_shap = []
    for run_index in range(num_runs):
        print(f"\n🔁 Run {run_index + 1}/{num_runs}")
        start_time = time.time()

        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + run_index,
            n_jobs=-1,
            eval_metric="logloss"
        )
        model.fit(X_df.values, y_array)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_df)

        explain_scaffold = df['scaffold'].values
        per_sample_mean = shap_values.mean(axis=1)
        all_scaffold_shap.extend(list(zip(explain_scaffold, per_sample_mean)))

        elapsed = time.time() - start_time
        print(f"✅ SHAP done for {len(X_df)} molecules in {elapsed:.2f} seconds")
        print(f"📌 Unique scaffolds analyzed: {df['scaffold'].nunique()}")

    df_shap = pd.DataFrame(all_scaffold_shap, columns=["scaffold", "mean_shap"])
    df_summary = (df_shap.groupby("scaffold")["mean_shap"]
                  .mean().reset_index()
                  .sort_values(by="mean_shap", ascending=False))
    df_summary["effect"] = df_summary["mean_shap"].apply(lambda x: "positive" if x > 0 else "negative")
    df_summary.to_csv(f"{output_dir}/scaffold_shap_summary.csv", index=False)
    print(f"\n✅ Saved scaffold SHAP summary → {output_dir}/scaffold_shap_summary.csv")


if __name__ == "__main__":
    # Chạy FAST mode
    main(mode="fast", num_runs=1, background_size=200)
