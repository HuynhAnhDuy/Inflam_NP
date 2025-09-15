import os
import random
import numpy as np
import pandas as pd
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

# ==== 1. Set seed ====
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# ==== 2. Scaffold + ECFP ==== 
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
    # ECFP của molecule
    ecfp_mol = smiles_to_ecfp(smiles, n_bits)
    # ECFP của scaffold
    scaffold = get_scaffold(smiles)
    ecfp_scaffold = smiles_to_ecfp(scaffold, n_bits) if scaffold else np.zeros(n_bits)
    # Concatenate (molecule + scaffold)
    return np.concatenate([ecfp_mol, ecfp_scaffold])

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

# ==== 4. Load train/test dataset có sẵn ====
df_train = pd.read_csv("InFlam_full_x_train.csv")
df_test = pd.read_csv("InFlam_full_x_test.csv")

df_train = df_train.dropna(subset=["canonical_smiles", "Label"])
df_test = df_test.dropna(subset=["canonical_smiles", "Label"])
df_train["Label"] = df_train["Label"].astype(int)
df_test["Label"] = df_test["Label"].astype(int)

X_train = np.array([smiles_to_ecfp_with_scaffold(s) for s in df_train["canonical_smiles"]])
y_train = df_train["Label"].values
X_test = np.array([smiles_to_ecfp_with_scaffold(s) for s in df_test["canonical_smiles"]])
y_test = df_test["Label"].values

# Reshape cho BiLSTM
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# ==== 5. Train & evaluate 1 lần ====
def run_once(seed):
    set_seed(seed)

    model = build_model(X_train.shape[2])
    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=32,
        verbose=1
    )

    y_prob = model.predict(X_test).ravel()
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

# ==== 6. Chạy 3 lần ====
all_metrics = []
for seed in [42, 43, 44]:
    result = run_once(seed)
    all_metrics.append(result)
    print(f"🔁 Run {seed}: {result}")

# ==== 7. Mean ± SD ====
df_metrics = pd.DataFrame(all_metrics)
mean_metrics = df_metrics.mean()
std_metrics = df_metrics.std()

summary = {metric: [f"{mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}"]
           for metric in mean_metrics.index}
summary_df = pd.DataFrame(summary)

# Lưu file
out_file = "/home/andy/andy/Inflam_NP/Scaffold_identify/Scaffold_metrics/InFlam_full_BiLSTM_metrics_ecfp+scaffold_summary.csv"
summary_df.to_csv(out_file, index=False)

print("\n✅ Đã lưu file:", out_file)
print("\n📊 Kết quả mean ± SD:")
print(summary_df)
