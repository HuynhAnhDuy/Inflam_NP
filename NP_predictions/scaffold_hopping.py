import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# ====== CONFIG ======
FILE1 = "/home/andy/andy/Inflam_NP/NP_predictions/NPASS_x_train_rdkit.csv"   # original (15k compounds)
FILE2 = "/home/andy/andy/Inflam_NP/NP_predictions/NPASS_candidates_final_rdkit.csv"   # candidate (304 compounds)
OUTPUT = "scaffold_hopping_topN_rdkit.csv"
N_NEIGHBORS = 20
SIM_THRESHOLD = 0.5
# ====================

# Đọc fingerprint
df1 = pd.read_csv(FILE1)
df2 = pd.read_csv(FILE2)

# Lấy danh sách cột fingerprint
fp_cols = [c for c in df1.columns if c.startswith("RDKit")]

X1 = df1[fp_cols].values.astype(int)
X2 = df2[fp_cols].values.astype(int)

# Hàm scaffold
def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaf)

df1["scaffold"] = df1["canonical_smiles"].apply(get_scaffold)
df2["scaffold"] = df2["canonical_smiles"].apply(get_scaffold)

# Nearest Neighbors
nn = NearestNeighbors(n_neighbors=N_NEIGHBORS, metric="jaccard")
nn.fit(X1)

distances, indices = nn.kneighbors(X2)

results = []
for i, row2 in df2.iterrows():
    for k in range(N_NEIGHBORS):
        j = indices[i, k]
        sim = 1 - distances[i, k]
        if sim >= SIM_THRESHOLD:
            if df1.loc[j, "scaffold"] != row2["scaffold"]:
                results.append({
                    "smiles1": df1.loc[j, "canonical_smiles"],
                    "scaffold1": df1.loc[j, "scaffold"],
                    "source1": "file1 (original)",
                    "smiles2": row2["canonical_smiles"],
                    "scaffold2": row2["scaffold"],
                    "source2": "file2 (candidate)",
                    "similarity": sim
                })

df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT, index=False)

# ====== PRINT ======
print(f"Tìm thấy {len(df_results)} cặp scaffold hopping (similarity ≥ {SIM_THRESHOLD})")
print(f"Kết quả lưu ở: {OUTPUT}\n")
print("5 dòng đầu tiên của kết quả:")
print(df_results.head(5).to_string(index=False))
