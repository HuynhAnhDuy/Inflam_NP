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
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ==== 4. Main SHAP analysis pipeline ====
def main(num_runs=5):
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

    all_scaffold_shap = []

    for run_index in range(num_runs):
        print(f"🔁 Run {run_index + 1}/{num_runs}")

        # === Train model on full data ===
        model = build_model(input_dim=X_df.shape[1])
        model.fit(np.expand_dims(X_df.values, axis=1), y_array, epochs=20, batch_size=32, verbose=0)

        # === SHAP explain ===
        def predict_fn(x):
            return model.predict(x.reshape((x.shape[0], 1, x.shape[1])))

        background = X_df.sample(n=min(200, len(X_df)), random_state=100 + run_index)
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(background, nsamples=100)
        expected_value = explainer.expected_value

        shap_values = shap_values[0] if isinstance(shap_values, list) else shap_values
        shap_values = np.array(shap_values).squeeze()

        background_scaffold = df.iloc[background.index]['scaffold'].values
        for i, scaffold in enumerate(background_scaffold):
            shap_score = shap_values[i].mean()
            all_scaffold_shap.append((scaffold, shap_score))

    # === Scaffold-level SHAP summary ===
    df_shap = pd.DataFrame(all_scaffold_shap, columns=["scaffold", "mean_shap"])
    df_summary = df_shap.groupby("scaffold")["mean_shap"].mean().reset_index()
    df_summary = df_summary.sort_values(by="mean_shap", ascending=False)
    df_summary["effect"] = df_summary["mean_shap"].apply(lambda x: "positive" if x > 0 else "negative")
    df_summary.to_csv(f"{output_dir}/scaffold_shap_summary.csv", index=False)
    print(f"✅ Saved scaffold SHAP summary → {output_dir}/scaffold_shap_summary.csv")

if __name__ == "__main__":
    main(num_runs=5)