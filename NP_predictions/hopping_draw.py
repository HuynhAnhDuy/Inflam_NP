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

df_all = df.dropna(subset=['canonical_smiles2']).drop_duplicates(subset=['canonical_smiles2'])
smiles_list = df_all[['canonical_smiles2', 'smiles2_name_ID']].values[:50]

# === 2. Khai báo scaffold templates chuẩn ===
templates = {
    "flavonoid": Chem.MolFromSmiles("O=C1C=CC(=O)C2=CC=CC=C2O1"),   # flavone core
    "chalcone": Chem.MolFromSmiles("O=C(/C=C/C1=CC=CC=C1)C2=CC=CC=C2")  # chalcone
}
for tpl in templates.values():
    rdDepictor.Compute2DCoords(tpl)

# === 3. SMARTS cho flavonoid core (nhiều cách viết để match linh hoạt)
flavonoid_cores = [
    Chem.MolFromSmarts("O=C1C=CC(=O)C2=CC=CC=C2O1"),   # kekulized
    Chem.MolFromSmarts("O=C1C=CC(=O)c2ccccc2O1")       # aromatic
]

# === 4. Hàm vẽ SMILES với scaffold được tô màu ===
def draw_smiles_with_highlighted_scaffold(
    smiles,
    legend_text,
    file_name,
    output_dir="/home/andy/andy/Inflam_NP/NP_predictions/structures_output",
    img_size=(500, 250)
):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"⚠️ Invalid SMILES: {smiles}")
        return

    # Nhận diện template
    template_used = None

    # Thử so SMARTS flavonoid
    for core in flavonoid_cores:
        if core is not None and mol.HasSubstructMatch(core):
            template_used = templates["flavonoid"]
            break

    # Nếu SMARTS không nhận, fallback bằng tên chứa 'flavone'
    if template_used is None and "flavone" in legend_text.lower():
        template_used = templates["flavonoid"]

    # Check chalcone
    if template_used is None and mol.HasSubstructMatch(templates["chalcone"]):
        template_used = templates["chalcone"]

    # Áp dụng orientation với atomMap
    if template_used:
        match = mol.GetSubstructMatch(template_used)
        if match:
            rdDepictor.GenerateDepictionMatching2DStructure(
                mol,
                template_used,
                atomMap=list(enumerate(match))
            )
        else:
            AllChem.Compute2DCoords(mol)
    else:
        AllChem.Compute2DCoords(mol)

    # Lấy scaffold thực tế để highlight
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    scaffold = Chem.MolFromSmiles(scaffold_smiles)

    match_atoms = mol.GetSubstructMatch(scaffold)
    if not match_atoms:
        print(f"⚠️ No scaffold match for: {smiles}")
        return

    atom_set = set(match_atoms)
    match_bonds = [bond.GetIdx() for bond in mol.GetBonds()
                   if bond.GetBeginAtomIdx() in atom_set and bond.GetEndAtomIdx() in atom_set]

    # Highlight màu xanh dương đậm
    color = (0, 0.6, 0)
    bond_colors = {b: color for b in match_bonds}

    # Vẽ SVG
    os.makedirs(output_dir, exist_ok=True)
    drawer = rdMolDraw2D.MolDraw2DSVG(*img_size)
    drawer.drawOptions().clearBackground = False
    drawer.drawOptions().highlightBondWidthMultiplier = 4
    drawer.drawOptions().fillHighlights = True
    drawer.drawOptions().highlightColour = color
    drawer.drawOptions().legendFontSize = 20

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

    print(f"✅ Saved: {file_name}.svg (legend = {legend_text}, template = {template_used is not None})")

# === 5. Chạy toàn bộ ===
for i, (smiles, compound_name) in enumerate(smiles_list, start=1):
    file_name = f"C_{i}"
    draw_smiles_with_highlighted_scaffold(smiles, compound_name, file_name)

print("🎯 Done: Flavonoid và Chalcone được nhận diện & align bằng template (SMARTS + fallback tên + atomMap).")
