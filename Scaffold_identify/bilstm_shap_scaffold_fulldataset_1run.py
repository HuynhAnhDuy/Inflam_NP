import os
import random
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from datetime import datetime
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
import time

# ==== 1. Set seed for reproducibility ====
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
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
        return np.array(fp)
    return np.zeros(n_bits)

# ==== 3. Build BiLSTM model ====
def build_model(input_dim):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, input_shape=(1, input_dim))),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(100, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# ==== 4. Main SHAP analysis pipeline ====
def main(mode="fast", num_runs=5, background_size=200, nsamples=100):
    """
    mode="fast": 1 run, explain toàn bộ dataset
    mode khác: giữ nguyên logic cũ với num_runs lần
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

    output_dir = f"shap_scaffold_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # ---- FAST MODE: single run over the full dataset ----
    if mode == "fast":
        run_index = 0
        print(f"\n🚀 FAST mode: single run on full dataset ({len(X_df)} molecules)")
        start_time = time.time()

        model = build_model(input_dim=X_df.shape[1])
        model.fit(np.expand_dims(X_df.values, axis=1),
                  y_array, epochs=20, batch_size=32, verbose=0)

        def predict_fn(x):
            return model.predict(x.reshape((x.shape[0], 1, x.shape[1])), verbose=0)

        background = X_df.sample(n=min(background_size, len(X_df)),
                                 random_state=100 + run_index)
        explain_data = X_df  # full dataset
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(explain_data, nsamples=nsamples)
        expected_value = explainer.expected_value

        shap_values = shap_values[0] if isinstance(shap_values, list) else shap_values
        shap_values = np.array(shap_values).squeeze()

        # Map đúng scaffold theo index của explain_data
        explain_scaffold = df.loc[explain_data.index, 'scaffold'].values

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
        print(f"✅ SHAP done for {len(explain_data)} molecules in {elapsed:.2f} seconds")
        print(f"📌 Unique scaffolds analyzed: {df['scaffold'].nunique()}")
        print(f"💾 Saved per-sample → {output_dir}/scaffold_shap_per_sample.csv")
        print(f"💾 Saved summary   → {output_dir}/scaffold_shap_summary.csv")
        return

    # ---- Legacy multi-run path (giữ nguyên nếu cần dùng) ----
    all_scaffold_shap = []
    for run_index in range(num_runs):
        print(f"\n🔁 Run {run_index + 1}/{num_runs}")
        start_time = time.time()

        model = build_model(input_dim=X_df.shape[1])
        model.fit(np.expand_dims(X_df.values, axis=1),
                  y_array, epochs=20, batch_size=32, verbose=0)

        def predict_fn(x):
            return model.predict(x.reshape((x.shape[0], 1, x.shape[1])), verbose=0)

        background = X_df.sample(n=min(background_size, len(X_df)),
                                 random_state=100 + run_index)
        explain_data = X_df

        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(explain_data, nsamples=nsamples)
        expected_value = explainer.expected_value

        shap_values = shap_values[0] if isinstance(shap_values, list) else shap_values
        shap_values = np.array(shap_values).squeeze()

        explain_scaffold = df.loc[explain_data.index, 'scaffold'].values
        per_sample_mean = shap_values.mean(axis=1)
        all_scaffold_shap.extend(list(zip(explain_scaffold, per_sample_mean)))

        elapsed = time.time() - start_time
        print(f"✅ SHAP done for {len(explain_data)} molecules in {elapsed:.2f} seconds")
        print(f"📌 Unique scaffolds analyzed: {df['scaffold'].nunique()}")

    df_shap = pd.DataFrame(all_scaffold_shap, columns=["scaffold", "mean_shap"])
    df_summary = (df_shap.groupby("scaffold")["mean_shap"]
                  .mean().reset_index()
                  .sort_values(by="mean_shap", ascending=False))
    df_summary["effect"] = df_summary["mean_shap"].apply(lambda x: "positive" if x > 0 else "negative")
    df_summary.to_csv(f"{output_dir}/scaffold_shap_summary.csv", index=False)
    print(f"\n✅ Saved scaffold SHAP summary → {output_dir}/scaffold_shap_summary.csv")

if __name__ == "__main__":
    # Chạy FAST mode: 1 run, explain toàn bộ dataset
    main(mode="fast", num_runs=1, background_size=200, nsamples=100)
