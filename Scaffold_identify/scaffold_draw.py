from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import pandas as pd
import os

# ========= CẤU HÌNH =========
CSV_PATH = "/home/andy/andy/Inflam_NP/Scaffold_identify/shap_scaffold_analysis_20250904_150634/scaffold_shap_summary.csv"   # <-- Đổi thành đường dẫn file CSV của bạn
SCAFFOLD_COL = "scaffold"             # <-- Đặt đúng tên cột trong CSV
N = 10                                # Số scaffold muốn vẽ
OUT_DIR = "Scaffold_structure"        # Thư mục output
IMG_SIZE = (500, 250)                 # Kích thước SVG (width, height)

# ========= ĐỌC CSV =========
df = pd.read_csv(CSV_PATH)
scaffolds = (
    df[SCAFFOLD_COL]
    .dropna()
    .astype(str)
    .drop_duplicates()
    .head(N)
    .tolist()
)

# ========= TẠO THƯ MỤC OUTPUT =========
os.makedirs(OUT_DIR, exist_ok=True)

# ========= VẼ & LƯU SVG =========
for i, scaffold_smiles in enumerate(scaffolds, start=1):
    mol = Chem.MolFromSmiles(scaffold_smiles)
    if mol is None:
        print(f"⚠️  Bỏ qua (SMILES không hợp lệ): {scaffold_smiles}")
        continue

    Chem.rdDepictor.Compute2DCoords(mol)

    w, h = IMG_SIZE
    drawer = rdMolDraw2D.MolDraw2DSVG(w, h)
    drawer.drawOptions().legendFontSize = 18
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    out_path = os.path.join(OUT_DIR, f"pos{i}.svg")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(svg)

    print(f"✅ Đã lưu: {out_path}")

print("🎉 Hoàn tất vẽ top scaffold đầu tiên.")
