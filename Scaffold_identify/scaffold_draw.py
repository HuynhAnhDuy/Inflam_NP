from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import pandas as pd
import os

# ========= CẤU HÌNH =========
CSV_PATH = "/home/andy/andy/Inflam_NP/Scaffold_identify/shap_scaffold_split_XGB_20250923_131651/scaffold_positive_overlap.csv"   # <-- CSV input
ID_COL = "ID"
SCAFFOLD_COL = "scaffold"
SHAP_COL = "mean_shap"

N = 15                                # số scaffold muốn vẽ
OUT_DIR = "/home/andy/andy/Inflam_NP/Scaffold_identify/shap_scaffold_split_XGB_20250923_131651"
IMG_SIZE = (500, 250)                 # (width, height) cho ảnh SVG

# ========= ĐỌC CSV =========
df = pd.read_csv(CSV_PATH)

scaffolds = (
    df[[ID_COL, SCAFFOLD_COL, SHAP_COL]]
    .dropna()
    .drop_duplicates(subset=[SCAFFOLD_COL])
    .head(N)
)

# ========= TẠO THƯ MỤC OUTPUT =========
os.makedirs(OUT_DIR, exist_ok=True)

# ========= VẼ & LƯU SVG =========
for i, row in scaffolds.iterrows():
    scaffold_smiles = str(row[SCAFFOLD_COL])
    mol_id = row[ID_COL]
    mean_shap = row[SHAP_COL]

    mol = Chem.MolFromSmiles(scaffold_smiles)
    if mol is None:
        print(f"⚠️  Bỏ qua (SMILES không hợp lệ): {scaffold_smiles}")
        continue

    Chem.rdDepictor.Compute2DCoords(mol)

    w, h = IMG_SIZE
    drawer = rdMolDraw2D.MolDraw2DSVG(w, h)
    drawer.drawOptions().legendFontSize = 18

    # Thêm legend: "ID - mean SHAP: ..."
    legend = f"{mol_id} - mean SHAP: {mean_shap:.5f}"

    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, legend=legend)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    out_path = os.path.join(OUT_DIR, f"scaffold_{mol_id}.svg")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(svg)

    print(f"✅ Đã lưu: {out_path}")

print("🎉 Hoàn tất vẽ scaffold với chú thích ID + SHAP.")
