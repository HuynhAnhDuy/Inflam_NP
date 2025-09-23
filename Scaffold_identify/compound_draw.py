import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

# ============= CONFIG =============
INPUT_FILE   = "/home/andy/andy/Inflam_NP/Scaffold_identify/shap_scaffold_split_XGB_20250923_131651/XGB_shap_compounds_LRo5_SA_2.csv"
SMILES_COL   = "canonical_smiles"
NAME_COL     = "Name"            # sẽ hiển thị dưới hình
OUT_DIR      = "./chem_svgs"
# ==================================

def smiles_to_mol(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        rdDepictor.Compute2DCoords(mol)
    return mol

def draw_mol_to_svg(mol, legend: str, size=(300, 250)) -> str:
    w, h = size
    drawer = rdMolDraw2D.MolDraw2DSVG(w, h)
    opts = drawer.drawOptions()
    opts.clearBackground = False   # 🔑 không vẽ nền trắng
    opts.addStereoAnnotation = True
    opts.explicitMethyl = True
    opts.fixedBondLength = 25
    opts.padding = 0.05
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, legend=legend)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()

def save_svg(svg_text: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg_text)

def main():
    # Nếu gặp lỗi UTF-8 thì thay bằng encoding="latin-1"
    df = pd.read_csv(INPUT_FILE, encoding="latin-1")

    os.makedirs(OUT_DIR, exist_ok=True)

    counter = 1
    for _, row in df.iterrows():
        smi = row.get(SMILES_COL, None)
        name = str(row.get(NAME_COL, f"compound_{counter}"))  # hiện dưới hình

        mol = smiles_to_mol(smi)
        if not mol:
            continue

        svg = draw_mol_to_svg(mol, legend=name, size=(350, 200))
        out_file = os.path.join(OUT_DIR, f"SHAP_compound_{counter}.svg")
        save_svg(svg, out_file)

        print(f"💾 Saved: {out_file}")
        counter += 1

if __name__ == "__main__":
    main()
