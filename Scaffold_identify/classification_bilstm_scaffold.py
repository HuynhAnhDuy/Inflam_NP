import os
import random
import numpy as np
import pandas as pd
from datetime import datetime
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem, DataStructs
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix
)

# === 0. Output folder ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"bilstm_results_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# === 1. Set seed ===
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# === 2. Scaffold + ECFP ===
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

def smiles_to_ecfp_with_scaffold(smiles, n_bits=2048):
    ecfp_mol = smiles_to_ecfp(smiles, n_bits)
    scaffold = get_scaffold(smiles)
    ecfp_scaffold = smiles_to_ecfp(scaffold, n_bits) if scaffold else np.zeros(n_bits)
    return np.concatenate([ecfp_mol, ecfp_scaffold])

def smiles_to_scaffold_only(smiles, n_bits=2048):
    scaffold = get_scaffold(smiles)
    return smiles_to_ecfp(scaffold, n_bits) if scaffold else np.zeros(n_bits)

# === 3. Build BiLSTM model ===
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

# === 4. Load train/test dataset ===
df_train = pd.read_csv("InFlam_full_x_train.csv").dropna(subset=["canonical_smiles", "Label"])
df_test = pd.read_csv("InFlam_full_x_test.csv").dropna(subset=["canonical_smiles", "Label"])
df_train["Label"] = df_train["Label"].astype(int)
df_test["Label"] = df_test["Label"].astype(int)

# Tạo features
X_train_ecfp_scaffold = np.array([smiles_to_ecfp_with_scaffold(s) for s in df_train["canonical_smiles"]])
X_test_ecfp_scaffold = np.array([smiles_to_ecfp_with_scaffold(s) for s in df_test["canonical_smiles"]])

X_train_scaffold_only = np.array([smiles_to_scaffold_only(s) for s in df_train["canonical_smiles"]])
X_test_scaffold_only = np.array([smiles_to_scaffold_only(s) for s in df_test["canonical_smiles"]])

y_train = df_train["Label"].values
y_test = df_test["Label"].values

# Reshape cho BiLSTM
X_train_ecfp_scaffold = X_train_ecfp_scaffold.reshape((X_train_ecfp_scaffold.shape[0], 1, X_train_ecfp_scaffold.shape[1]))
X_test_ecfp_scaffold = X_test_ecfp_scaffold.reshape((X_test_ecfp_scaffold.shape[0], 1, X_test_ecfp_scaffold.shape[1]))

X_train_scaffold_only = X_train_scaffold_only.reshape((X_train_scaffold_only.shape[0], 1, X_train_scaffold_only.shape[1]))
X_test_scaffold_only = X_test_scaffold_only.reshape((X_test_scaffold_only.shape[0], 1, X_test_scaffold_only.shape[1]))

# === 5. Train + Evaluate ===
def train_bilstm(X_train, y_train, X_test, y_test, seed):
    set_seed(seed)
    model = build_model(X_train.shape[2])
    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=32,
        verbose=0
    )
    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob > 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "AUROC": roc_auc_score(y_test, y_prob),
        "AUPRC": average_precision_score(y_test, y_prob),
        "Sensitivity": sensitivity,
        "Specificity": specificity
    }

# === 6. Run nhiều seed cho từng loại features ===
def run_experiment(X_train, y_train, X_test, y_test, name, output_dir):
    results = []
    for run, seed in enumerate([42, 43, 44], start=1):
        metrics = train_bilstm(X_train, y_train, X_test, y_test, seed)
        metrics["Run"] = run
        metrics["Seed"] = seed
        results.append(metrics)
        print(f"🔁 {name} - Run {run} (Seed={seed}): {metrics}")

    df_raw = pd.DataFrame(results)
    df_raw.to_csv(os.path.join(output_dir, f"bilstm_metrics_{name}_raw.csv"), index=False)

    metrics_order = ["Accuracy", "MCC", "AUROC", "AUPRC", "Sensitivity", "Specificity"]
    df_summary_stats = df_raw[metrics_order].agg(['mean', 'std']).T
    df_summary_stats[name] = (
        df_summary_stats["mean"].round(3).astype(str) + " ± " + df_summary_stats["std"].round(3).astype(str)
    )
    df_summary = df_summary_stats[[name]].T
    df_summary.index = [name]
    df_summary.to_csv(os.path.join(output_dir, f"bilstm_metrics_{name}_summary.csv"))
    return df_summary

# === 7. Chạy 2 mô hình ===
summary_ecfp_scaffold = run_experiment(X_train_ecfp_scaffold, y_train, X_test_ecfp_scaffold, y_test, "ecfp+scaffold", output_dir)
summary_scaffold_only = run_experiment(X_train_scaffold_only, y_train, X_test_scaffold_only, y_test, "scaffold_only", output_dir)

# === 8. Gộp kết quả ===
df_summary_all = pd.concat([summary_ecfp_scaffold, summary_scaffold_only])
df_summary_all.to_csv(os.path.join(output_dir, "bilstm_metrics_all_summary.csv"))

print(f"\n📊 Trung bình ± SD trên 3 lần chạy (BiLSTM):")
print(df_summary_all)
print(f"\n📁 Tất cả kết quả được lưu tại: {output_dir}")
