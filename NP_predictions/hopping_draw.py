import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdDepictor
from rdkit.Chem import AllChem
import os

# === 1. Load dữ liệu chứa SMILES, loại trùng ===
df = pd.read_csv(
    "/home/andy/andy/Inflam_NP/NP_predictions/NPASS_common_scaffold_hopping_annotated.csv",
    encoding='latin-1'
)

# Drop_duplicates (nếu cần)
df_all = df.dropna(subset=['canonical_smiles2']).drop_duplicates(subset=['canonical_smiles2'])

# Lấy tối đa 50 mẫu
smiles_list = df_all[['canonical_smiles2', 'smiles2_name_ID']].values[:50]

# === 2. Vẽ SMILES với scaffold được tô màu và thêm chú thích ===
def draw_smiles_with_highlighted_scaffold(
    smiles,
    legend_text,   # dùng để hiển thị dưới hình
    file_name,     # dùng làm tên file SVG
    output_dir="/home/andy/andy/Inflam_NP/NP_predictions/structures_output",
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

    # === Highlight scaffold ===
    color = (0, 0, 0)  # đen
    bond_colors = {b: color for b in match_bonds}

    os.makedirs(output_dir, exist_ok=True)

    drawer = rdMolDraw2D.MolDraw2DSVG(*img_size)
    drawer.drawOptions().clearBackground = False
    drawer.drawOptions().highlightBondWidthMultiplier = 4
    drawer.drawOptions().fillHighlights = True
    drawer.drawOptions().highlightColour = color
    drawer.drawOptions().legendFontSize = 20  # chỉnh cỡ chữ chú thích

    # legend = tên hợp chất (smiles1_name)
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightBonds=match_bonds,
        highlightBondColors=bond_colors,
        legend=legend_text
    )
    drawer.FinishDrawing()

    svg = drawer.GetDrawingText()
    out_path = os.path.join(output_dir, f"{file_name}.svg")
    with open(out_path, "w") as f:
        f.write(svg)

    print(f"✅ Saved: {file_name}.svg (legend = {legend_text})")

# === 3. Chạy cho toàn bộ danh sách SMILES duy nhất ===
for i, (smiles, compound_name) in enumerate(smiles_list, start=1):
    file_name = f"C_{i}"  # tên file chỉ số thứ tự
    draw_smiles_with_highlighted_scaffold(smiles, compound_name, file_name)

print("🎯 Done: All scaffold bonds are highlighted in color with legends (unique SMILES only).")
