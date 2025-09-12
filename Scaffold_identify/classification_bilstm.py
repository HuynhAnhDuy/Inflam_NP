import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, matthews_corrcoef,
    roc_auc_score, accuracy_score
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
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

# === 6. Reshape for BiLSTM ===
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# === 7. Function to build and train model ===
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def build_and_train_model(X_train, y_train, X_test, seed):
    set_seed(seed)
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, input_shape=(1, X_train.shape[2]))),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(100, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1, verbose=0)
    y_proba = model.predict(X_test).flatten()
    y_pred = (y_proba >= 0.5).astype(int)
    return y_pred, y_proba

# === 8. Run 3 independent training rounds ===
metrics = {
    'accuracy': [], 'mcc': [], 'sensitivity': [], 'specificity': [],'auc' : []
}

for seed in [42, 43, 44]:
    y_pred, y_proba = build_and_train_model(X_train, y_train, X_test, seed)
    conf_matrix = confusion_matrix(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['accuracy'].append(acc)
    metrics['mcc'].append(mcc)    
    metrics['specificity'].append(specificity)
    metrics['sensitivity'].append(sensitivity)
    metrics['auc'].append(auc)

# === 9. Print mean ± std for each metric ===
print("\n📊 Mean ± SD over 3 runs:")
for name, values in metrics.items():
    mean = np.mean(values)
    std = np.std(values)
    print(f"{name.capitalize():<12}: {mean:.3f} ± {std:.3f}")
