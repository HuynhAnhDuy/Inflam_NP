import pandas as pd
import numpy as np
import re
from rdkit import Chem
from rdkit.Chem import AllChem
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# === 1. Đọc dữ liệu CSV đã gán nhãn (0/1) ===
df = pd.read_csv("rgroup_labeled_majority.csv")  # hoặc rgroup_labeled_majority.csv
df = df.dropna(subset=["rgroup_smiles", "Label"])

print(f"👉 Tổng số R-group: {len(df)}")

# === 2. Tiền xử lý R-group ===
def strip_attachment(smi: str) -> str:
    """Bỏ [*:n] khỏi SMILES"""
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

# === 3. Train-test split (chỉ để đánh giá, không dùng cho SHAP) ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === 4. Train XGBoost trên toàn bộ dữ liệu ===
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
model.fit(X, y)

print("\n📊 Hiệu năng (train/test split để tham khảo):")
print(classification_report(y_test, model.predict(X_test)))

# === 5. Tính SHAP values trên toàn bộ dataset ===
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Với classification 2 lớp: shap_values[1] là lớp "1" (active)
if isinstance(shap_values, list):  
    shap_values = shap_values[1]

# === 6. Trung bình SHAP theo (R-label, R-group) ===
df = df.reset_index(drop=True)
df["mean_shap"] = shap_values.mean(axis=1)

summary_by_label = (
    df.groupby(["rgroup_label", "clean_smi"])["mean_shap"]
    .mean()
    .reset_index()
    .sort_values("mean_shap", ascending=False)
)

# Thêm cột đánh giá Positive / Negative
summary_by_label["effect"] = summary_by_label["mean_shap"].apply(
    lambda x: "Positive" if x > 0 else "Negative"
)

# Ghi ra CSV
summary_by_label.to_csv("rgroup_shap_xgb_majority_full.csv", index=False)

print("\n✅ Đã lưu kết quả SHAP (toàn bộ dataset) vào: rgroup_shap_xgb_enrichment_full.csv")
print("🔎 Top R-group Positive:")
print(summary_by_label[summary_by_label["effect"]=="Positive"].head(10))
print("\n🔎 Top R-group Negative:")
print(summary_by_label[summary_by_label["effect"]=="Negative"].head(10))
