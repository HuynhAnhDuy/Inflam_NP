import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ============== CONFIG ==============
INPUT_CSV = "/home/andy/andy/Inflam_NP/Scaffold_clustering/scaffold_shap_summary_test.csv"
OUTPUT_EMBED = "scaffolds_tsne_2d.csv"
OUTPUT_FIG  = "tsne_mean_shap.svg"

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

# 4. Run t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    n_iter_without_progress=300,
    init="random",
    metric="euclidean",
    random_state=42
)

embedding = tsne.fit_transform(X)
df_valid["tsne_x"] = embedding[:, 0]
df_valid["tsne_y"] = embedding[:, 1]

# Save CSV
df_valid.to_csv(OUTPUT_EMBED, index=False)
print("Saved coordinates:", OUTPUT_EMBED)

# 5. Plot t-SNE colored by mean SHAP
plt.figure(figsize=(9, 6))
sc = plt.scatter(
    df_valid["tsne_x"],
    df_valid["tsne_y"],
    c=df_valid[SHAP_COL],
    s=12,
    alpha=0.85,
    cmap="viridis"
)

plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE projection colored by mean SHAP")
plt.colorbar(sc, label=SHAP_COL)
plt.tight_layout()

# Save to SVG
plt.savefig(OUTPUT_FIG, format="svg")
print("Saved SVG figure:", OUTPUT_FIG)
