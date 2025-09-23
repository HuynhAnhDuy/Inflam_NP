# pip install rdkit-pypi pandas

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

# === Load CSV ===
df = pd.read_csv("XGB_Rgroup_overall_top_positive.csv")

# Lấy toàn bộ R_group_smiles + mean_shap
rgroups = df[["R_group_smiles", "mean_shap"]].dropna().drop_duplicates()

# Convert SMILES -> Mol
mols = []
labels = []
for smi, shap_val in zip(rgroups["R_group_smiles"], rgroups["mean_shap"]):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        Chem.rdDepictor.Compute2DCoords(mol)
        mols.append(mol)
        labels.append(f"{smi}\nSHAP={shap_val:.3f}")

# Vẽ grid SVG
img = Draw.MolsToGridImage(
    mols,
    molsPerRow=7,
    subImgSize=(150,200),
    legends=labels,
    useSVG=True
)

# Lưu ra file SVG
with open("XGB_All_Rgroups_with_SHAP.svg", "w", encoding="utf-8") as f:
    f.write(img)

print("Saved All_Rgroups_with_SHAP.svg")
