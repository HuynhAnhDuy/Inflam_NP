import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdDepictor
from rdkit.Chem import AllChem
import os

# === 1. Load dữ liệu chứa SMILES ===
df = pd.read_csv(
    "/home/andy/andy/Inflam_NP/Scaffold_identify/shap_scaffold_split_XGB_20250915_151119/molecules_scaffold_O=c1c2ccccc2oc2ccccc12.csv",
    encoding='latin-1'
)
smiles_list = df['canonical_smiles'].dropna().unique()[:50]

# === 2. Vẽ SMILES với scaffold được tô màu ===
def draw_smiles_with_highlighted_scaffold(
    smiles,
    name,
    output_dir="/home/andy/andy/Inflam_NP/Scaffold_identify/shap_scaffold_split_XGB_20250915_151119",
    img_size=(500, 250)
):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"⚠️ Invalid SMILES: {smiles}")
        return

    AllChem.Compute2DCoords(mol)

    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    scaffold = Chem.MolFromSmiles(scaffold_smiles)

    match_atoms = mol.GetSubstructMatch(scaffold)
    if not match_atoms:
        print(f"⚠️ No scaffold match for: {smiles}")
        return

    atom_set = set(match_atoms)
    match_bonds = []
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in atom_set and a2 in atom_set:
            match_bonds.append(bond.GetIdx())

    # === Dùng màu cam ambient ===
    color = (0.1, 0.6, 1.0)
    bond_colors = {b: color for b in match_bonds}

    os.makedirs(output_dir, exist_ok=True)

    drawer = rdMolDraw2D.MolDraw2DSVG(*img_size)
    drawer.drawOptions().clearBackground = True
    drawer.drawOptions().highlightBondWidthMultiplier = 4
    drawer.drawOptions().fillHighlights = True
    drawer.drawOptions().highlightColour = color # fallback nếu không truyền bond_colors

    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightBonds=match_bonds,
        highlightBondColors=bond_colors
    )
    drawer.FinishDrawing()

    svg = drawer.GetDrawingText()
    with open(os.path.join(output_dir, f"{name}.svg"), "w") as f:
        f.write(svg)

    print(f"✅ Saved: {name}.svg with scaffold bonds highlighted in color")

# === 3. Chạy cho toàn bộ danh sách SMILES ===
for i, smiles in enumerate(smiles_list):
    name = f"compound_{i+1}"
    draw_smiles_with_highlighted_scaffold(smiles, name)

print("🎯 Done: All scaffold bonds are highlighted in color.")
