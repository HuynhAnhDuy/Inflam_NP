import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OneHotEncoder

# === Load data ===
df_train = pd.read_csv("capsule_x_train.csv")
df_test = pd.read_csv("irac_2b.csv")

# === Scaffold & fingerprint functions ===
def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    return "None"

def mol_to_ecfp(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return np.zeros((nBits,))
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    return np.array(fp)

# === Tính scaffold + ECFP ===
for df in [df_train, df_test]:
    df['scaffold'] = df['canonical_smiles'].apply(get_scaffold)
    df['ecfp'] = df['canonical_smiles'].apply(mol_to_ecfp)

X_ecfp_train = np.stack(df_train['ecfp'].values)
X_ecfp_test = np.stack(df_test['ecfp'].values)

# === Encode scaffold ===
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_scaffold_train = ohe.fit_transform(df_train[['scaffold']])
X_scaffold_test = ohe.transform(df_test[['scaffold']])

# === Combine features ===
X_train = np.hstack([X_ecfp_train, X_scaffold_train])
X_test = np.hstack([X_ecfp_test, X_scaffold_test])
y_train = df_train['Toxicity Value'].values

# === Chạy mô hình 3 lần và lấy xác suất ===
prob_matrix = []
for i in range(3):
    model = LGBMClassifier(class_weight='balanced', random_state=42 + i)
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    prob_matrix.append(y_proba)

# === Tính xác suất trung bình ===
mean_probs = np.mean(prob_matrix, axis=0)

# === Tạo output ===
result_df = df_test[['canonical_smiles']].copy()
result_df['Scaffold_avg'] = mean_probs
result_df['predicted_label'] = (mean_probs >= 0.5).astype(int)

# === Xuất kết quả ===
result_df.to_csv("irac_2b_prediction_probabilities_3runs.csv", index=False)
print("✅ Đã lưu file: prediction_probabilities_3runs.csv")
