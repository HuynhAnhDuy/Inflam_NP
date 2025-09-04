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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import random
import os

# === 1. Load data ===
df_train = pd.read_csv("InFlam_full_x_train.csv")
df_test = pd.read_csv("InFlam_full_x_test.csv")

# === 2. Scaffold & fingerprint functions ===
def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    return "None"

def mol_to_ecfp(smiles, radius=2, nBits=512):  # Giảm xuống 512-bit
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

# === 6. Reshape for BiLSTM ===
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# === 7. Seed control ===
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# === 8. Build and train BiLSTM ===
def build_and_train_model(X_train, y_train, X_test, seed):
    set_seed(seed)
    model = Sequential([
        Input(shape=(1, X_train.shape[2])),
        Bidirectional(LSTM(32, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(16)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1, verbose=0)
    y_proba = model.predict(X_test).flatten()
    y_pred = (y_proba >= 0.5).astype(int)
    return y_pred, y_proba

# === 9. Run 3 independent training rounds ===
results = []
for run, seed in enumerate([42, 43, 44], start=1):
    y_pred, y_proba = build_and_train_model(X_train, y_train, X_test, seed)
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

# === 10. Lưu kết quả từng run
df_raw = pd.DataFrame(results)
df_raw.to_csv("bilstm_metrics_ecfp+scaffold_raw.csv", index=False)
print("✅ Đã lưu kết quả từng lần chạy → bilstm_metrics_ecfp+scaffold_raw.csv")

# === 11. Tính mean ± std và lưu
metrics_order = [
    "Accuracy", "Balanced Accuracy", "AUROC", "AUPRC",
    "MCC", "Precision", "Sensitivity", "Specificity"
]

df_summary_stats = df_raw[metrics_order].agg(['mean', 'std']).T
df_summary_stats["Scaffold"] = (
    df_summary_stats["mean"].round(3).astype(str) + " ± " + df_summary_stats["std"].round(3).astype(str)
)

# Đưa về format: hàng = Scaffold, cột = các metrics
df_summary = df_summary_stats[["Scaffold"]].T
df_summary.index = ["Scaffold"]

df_summary.to_csv("bilstm_metrics_ecfp+scaffold_summary.csv")
print("✅ Đã lưu summary → bilstm_metrics_ecfp+scaffold_summary.csv")

# === 12. In terminal
print("\n📊 Trung bình ± SD trên 3 lần chạy (BiLSTM):")
print(df_summary)
