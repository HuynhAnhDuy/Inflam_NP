from meeko import MoleculePreparation, PDBQTWriterLegacy
from rdkit import Chem

# 1. Đọc file PDB từ PyMOL
mol = Chem.MolFromPDBFile("AF7_ref.pdb", removeHs=False)

# 2. Thêm và tường minh hóa toàn bộ nguyên tử Hydro
mol = Chem.AddHs(mol, addCoords=True)

# 3. Chuẩn bị ligand bằng cấu trúc mới của Meeko
preparator = MoleculePreparation()
setup_list = preparator.prepare(mol)

# 4. Ghi file .pdbqt (Lấy [0] từ kết quả trả về vì nó là tuple)
with open("AF7_ref.pdbqt", "w") as f:
    pdbqt_data = PDBQTWriterLegacy.write_string(setup_list[0])
    # Nếu pdbqt_data là tuple, ta lấy phần tử đầu tiên là chuỗi nội dung
    if isinstance(pdbqt_data, tuple):
        f.write(pdbqt_data[0])
    else:
        f.write(pdbqt_data)

print("[SUCCESS] Đã chuyển đổi thành công sang file 7DN_ref.pdbqt hoàn chỉnh!")