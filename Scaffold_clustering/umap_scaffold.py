import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from umap import UMAP          # 🔹 Đổi từ TSNE sang UMAP
import matplotlib.pyplot as plt

# ============== CONFIG ==============
INPUT_CSV = "/home/andy/andy/Inflam_NP/Scaffold_clustering/scaffold_shap_summary_test_TEST_SET.csv"
OUTPUT_EMBED = "scaffolds_umap_2d.csv"     # 🔹 Đổi tên file cho đúng với UMAP
OUTPUT_FIG  = "umap_mean_shap.svg"

SMILES_COL = "scaffold_smiles"
SHAP_COL   = "mean_shap"

FP_NBITS = 2048
FP_RADIUS = 4      # ECFP4
# ====================================

# 1. Read CSV
df = pd.read_csv(INPUT_CSV)

# 2. Parse SMILES
mols = []
valid_idx = []
for i, smi in enumerate(df[SMILES_COL].astype(str)):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        mols.append(mol)
        valid_idx.append(i)

df_valid = df.loc[valid_idx].reset_index(drop=True)
print("Valid scaffolds:", len(df_valid))

# 3. Generate ECFP4 fingerprints → numpy array
fps = []
for mol in mols:
    bv = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=FP_RADIUS, nBits=FP_NBITS
    )
    arr = np.zeros((FP_NBITS,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    fps.append(arr)

X = np.array(fps)
print("Fingerprint matrix shape:", X.shape)

# 4. Run UMAP (thay cho t-SNE)
umap = UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    random_state=42
)

embedding = umap.fit_transform(X)
df_valid["umap_x"] = embedding[:, 0]
df_valid["umap_y"] = embedding[:, 1]

# Save CSV
df_valid.to_csv(OUTPUT_EMBED, index=False)
print("Saved coordinates:", OUTPUT_EMBED)

# 5. Plot UMAP colored by mean SHAP (đổi colormap)
plt.figure(figsize=(8, 6))

sc = plt.scatter(
    df_valid["umap_x"],
    df_valid["umap_y"],
    c=df_valid[SHAP_COL],
    s=12,
    alpha=0.9,
    cmap="inferno"   # 🔹 Đổi màu: phù hợp khi SHAP có cả âm/dương
)

plt.xlabel("UMAP 1", fontsize=12, fontweight='bold', fontstyle='italic', family='sans-serif') 
plt.ylabel("UMAP 2", fontsize=12, fontweight='bold', fontstyle='italic', family='sans-serif') 
plt.title("UMAP visualization of positive scaffolds",
          fontsize=12, fontweight='bold', family='sans-serif')
plt.colorbar(sc, label="Mean SHAP values")
plt.tight_layout()

# Save to SVG
plt.savefig(OUTPUT_FIG, format="svg")
print("Saved SVG figure:", OUTPUT_FIG)
