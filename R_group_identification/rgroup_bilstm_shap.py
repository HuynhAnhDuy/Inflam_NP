import pandas as pd
import numpy as np
import re
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam

import shap

# === 1. Đọc dữ liệu CSV đã gán nhãn ===
df = pd.read_csv("rgroup_labeled_enrichment.csv")  # hoặc rgroup_labeled_enrichment.csv
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
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

df["clean_smi"] = df["rgroup_smiles"].apply(strip_attachment)
X = np.array([smiles_to_morgan(smi) for smi in df["clean_smi"]])
y = df["Label"].values

# Reshape cho BiLSTM (samples, timesteps=1, features)
X = X.reshape((X.shape[0], 1, X.shape[1]))

# === 3. Train-test split chỉ để đánh giá hiệu năng ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === 4. Build BiLSTM model ===
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

model = build_model(X.shape[2])

# === 5. Train model trên toàn bộ dữ liệu ===
history = model.fit(
    X, y,
    validation_split=0.2,
    epochs=10,  # tăng nếu dataset lớn
    batch_size=32,
    verbose=1
)

# Đánh giá trên split train/test để tham khảo
y_pred = (model.predict(X_test) > 0.5).astype(int)
print("\n📊 Hiệu năng trên tập test (tham khảo):")
print(classification_report(y_test, y_pred))

# === 6. SHAP GradientExplainer trên toàn bộ dataset ===
background = X[np.random.choice(X.shape[0], min(100, X.shape[0]), replace=False)]  # subset làm background

explainer = shap.GradientExplainer(model, background)
shap_values = explainer.shap_values(X)

# Nếu trả về list thì lấy phần tử đầu
if isinstance(shap_values, list):
    shap_values = shap_values[0]

# Bỏ chiều timesteps (N,1,2048) → (N,2048)
shap_values = shap_values.squeeze()

# === 7. Gom attribution theo R-group ===
df = df.reset_index(drop=True)
df["mean_shap"] = shap_values.mean(axis=1)

summary_by_label = (
    df.groupby(["rgroup_label", "clean_smi"])["mean_shap"]
    .mean()
    .reset_index()
    .sort_values("mean_shap", ascending=False)
)

# Thêm Positive / Negative
summary_by_label["effect"] = summary_by_label["mean_shap"].apply(
    lambda x: "Positive" if x > 0 else "Negative"
)

# === 8. Xuất CSV ===
summary_by_label.to_csv("rgroup_shap_bilstm_enrichment_full.csv", index=False)

print("\n✅ Đã lưu kết quả SHAP (toàn bộ dataset) cho BiLSTM")
print("🔎 Top R-group Positive:")
print(summary_by_label[summary_by_label['effect']=="Positive"].head(10))
print("\n🔎 Top R-group Negative:")
print(summary_by_label[summary_by_label['effect']=="Negative"].head(10))
