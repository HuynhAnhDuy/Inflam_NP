import pandas as pd
import numpy as np
import re
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam

# === 1. Đọc dữ liệu CSV đã gán nhãn ===
df = pd.read_csv("rgroup_labeled_majority.csv")
df = df.dropna(subset=["rgroup_smiles", "Label"])
df["Label"] = df["Label"].astype(int)
print(f"👉 Tổng số R-group: {len(df)}")

# === 2. Tiền xử lý SMILES (bỏ [*:n]) ===
def strip_attachment(smi: str) -> str:
    return re.sub(r"\[\*:[0-9]+\]", "", smi)

def smiles_to_morgan(smi, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros((nBits,))
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

df["clean_smi"] = df["rgroup_smiles"].apply(strip_attachment)
X = np.array([smiles_to_morgan(smi) for smi in df["clean_smi"]])
y = df["Label"].values

# Reshape cho BiLSTM
X = X.reshape((X.shape[0], 1, X.shape[1]))

# === 3. Build BiLSTM model ===
def build_model(input_dim):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(1, input_dim))))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.3))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# === 4. Hàm train & evaluate ===
def run_once(random_state):
    # Train-test split 80:20 (random_state thay đổi)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    model = build_model(X.shape[2])
    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=32,
        verbose=0
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

# === 5. Chạy 3 lần ===
all_metrics = []
for seed in [42, 43, 44]:  # 3 random_state khác nhau
    result = run_once(seed)
    all_metrics.append(result)
    print(f"🔁 Run với random_state={seed}: {result}")

# === 6. Tính mean ± SD ===
df_metrics = pd.DataFrame(all_metrics)
mean_metrics = df_metrics.mean()
std_metrics = df_metrics.std()

summary = {metric: [f"{mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}"] 
           for metric in mean_metrics.index}
summary_df = pd.DataFrame(summary)

# Lưu ra CSV
summary_df.to_csv("rgroup_labeled_majority_test_metrics.csv", index=False)

print("\n✅ Metrics trung bình ± SD đã được lưu vào file output")
print(summary_df)
