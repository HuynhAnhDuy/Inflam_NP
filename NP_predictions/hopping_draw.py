import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
import os

# === 1. Load dữ liệu chứa SMILES ===
df = pd.read_csv(
    "/home/andy/andy/Inflam_NP/NP_predictions/NPASS_common_scaffold_hopping.csv",
    encoding='latin-1'
).dropna(subset=["smiles1_can", "smiles2_can"])

pairs = df[["smiles1_can", "smiles2_can"]].drop_duplicates().values[:13]  # lấy 20 cặp đầu

# === 2. Hàm vẽ 1 SMILES với scaffold highlight ===
def draw_molecule_with_scaffold(smiles, name, output_dir, color=(0.1, 0.6, 1.0)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"⚠️ Invalid SMILES: {smiles}")
        return

    AllChem.Compute2DCoords(mol)

    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold:
        scaffold_smiles = Chem.MolToSmiles(scaffold)
        scaffold = Chem.MolFromSmiles(scaffold_smiles)

        match_atoms = mol.GetSubstructMatch(scaffold)
        atom_set = set(match_atoms)
        match_bonds = [
            bond.GetIdx() for bond in mol.GetBonds()
            if bond.GetBeginAtomIdx() in atom_set and bond.GetEndAtomIdx() in atom_set
        ]
    else:
        match_bonds = []

    drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
    drawer.drawOptions().clearBackground = True
    drawer.drawOptions().highlightBondWidthMultiplier = 4
    drawer.drawOptions().fillHighlights = True

    bond_colors = {b: color for b in match_bonds}

    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightBonds=match_bonds,
        highlightBondColors=bond_colors
    )
    drawer.FinishDrawing()

    svg = drawer.GetDrawingText()

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{name}.svg")
    with open(filepath, "w") as f:
        f.write(svg)

    print(f"✅ Saved: {filepath}")

# === 3. Vẽ cho từng cặp ===
OUTPUT_DIR = "/home/andy/andy/Inflam_NP/NP_predictions/NPASS_common_hopping"

for i, (smi1, smi2) in enumerate(pairs, 1):
    draw_molecule_with_scaffold(smi1, f"pair{i}_compound1", OUTPUT_DIR)
    draw_molecule_with_scaffold(smi2, f"pair{i}_compound2", OUTPUT_DIR)

print("🎯 Done: All compounds drawn with scaffold highlight.")
