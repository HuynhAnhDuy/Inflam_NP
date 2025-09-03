import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, matthews_corrcoef,
    roc_auc_score, accuracy_score
)

# === 1. Load data ===
df_train = pd.read_csv("InFlam_full_x_train.csv")
df_test = pd.read_csv("InFlam_full_x_test.csv")

# === 2. Scaffold & fingerprint functions ===
def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    return "None"

def mol_to_ecfp(smiles, radius=2, nBits=2048):
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

# === 6. Train Random Forest ===
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# === 7. Metrics ===
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
accuracy = accuracy_score(y_test, y_pred)

# Confusion matrix unpack
tn, fp, fn, tp = conf_matrix.ravel()

# Custom metrics
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # same as recall of class 1

# === 8. In kết quả ===
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print(f"\n✅ Accuracy:      {accuracy:.3f}")
print(f"✅ MCC:           {mcc:.3f}")
print(f"✅ Sensitivity:   {sensitivity:.3f}")
print(f"✅ Specificity:   {specificity:.3f}")
print(f"✅ AUC:           {auc:.3f}")
print(f"✅ Confusion Matrix:\n{conf_matrix}")
