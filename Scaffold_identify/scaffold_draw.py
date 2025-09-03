from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import os

# === Scaffold SMILES và tên ===
scaffold_smiles = "O=C1c2ccccc2CCc2cc(Nc3ccccc3)ccc21"
scaffold_name = "pos1"  # tên file không nên chứa ký tự đặc biệt

# === Tạo đối tượng phân tử ===
mol = Chem.MolFromSmiles(scaffold_smiles)
if mol is None:
    raise ValueError("SMILES không hợp lệ.")

# === Tối ưu cấu trúc để hiển thị đẹp ===
Chem.rdDepictor.Compute2DCoords(mol)

# === Thiết lập vẽ SVG ===
drawer = rdMolDraw2D.MolDraw2DSVG(500, 250)  # Kích thước ảnh (pixels)
drawer.drawOptions().legendFontSize = 18     # Cỡ chữ cho chú thích
drawer.DrawMolecule(mol)
drawer.FinishDrawing()

# === Lấy nội dung SVG và lưu vào file ===
svg = drawer.GetDrawingText()
output_file = f"{scaffold_name}.svg"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(svg)

print(f"✅ Đã lưu scaffold SVG tại: {output_file}")
