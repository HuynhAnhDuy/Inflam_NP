#!/usr/bin/env python3
import os
import sys
import re
import pandas as pd

try:
    from openbabel import openbabel
except ImportError:
    print("[ERROR] Không tìm thấy thư viện 'openbabel'.")
    sys.exit(1)

# ==========================
# Function helpers
# ==========================
def sanitize_filename(name):
    """
    Loại bỏ các ký tự đặc biệt không hợp lệ trong tên file của hệ điều hành
    và thay thế khoảng trắng bằng dấu gạch dưới.
    """
    name = str(name).strip().replace(" ", "_")
    # Chỉ giữ lại chữ cái, số, dấu gạch dưới, gạch ngang
    return re.sub(r'(?u)[^-\w.]', '', name)

def convert_smiles_to_pdbqt(smiles, output_pdbqt, ligand_name):
    """
    Chuyển SMILES trực tiếp sang PDBQT: Thêm Hydro, dựng 3D, 
    tối ưu hóa năng lượng (MMFF94) và tính điện tích Gasteiger.
    """
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("smi", "pdbqt")
    
    mol = openbabel.OBMol()
    
    # 1. Đọc chuỗi SMILES
    if not obConversion.ReadString(mol, smiles):
        print(f"[ERROR] Không thể đọc chuỗi SMILES của {ligand_name}")
        return False
        
    # 2. Thêm tất cả Hydro
    mol.AddHydrogens()
    
    # 3. Dựng cấu trúc hình học 3D thô
    builder = openbabel.OBBuilder()
    builder.Build(mol)
    
    # 4. Tối ưu hóa năng lượng cấu trúc (Energy Minimization)
    ff = openbabel.OBForceField.FindForceField("MMFF94")
    if ff is None:
        ff = openbabel.OBForceField.FindForceField("UFF")
        
    if ff is not None:
        ff.Setup(mol)
        ff.ConjugateGradients(500)  # Tối ưu hóa 500 bước
        ff.GetCoordinates(mol)
    else:
        print(f"[WARNING] Không tìm thấy Force Field để tối ưu hóa cho {ligand_name}!")

    # 5. Ghi trực tiếp ra file PDBQT
    success = obConversion.WriteFile(mol, output_pdbqt)
    if success:
        return True
    else:
        print(f"[FAILED] Ghi file PDBQT thất bại cho {ligand_name}")
        return False

# ==========================
# Main workflow
# ==========================
def main():
    # Thư mục gốc chứa file CSV của bạn
    base_dir = "D:\Andy\Inflam_NP\Docking_InFlam\compounds_original"
    csv_path = os.path.join(base_dir, "compounds_for_docking_SA2.csv")
    
    # Thư mục con chứa file PDBQT kết quả
    output_dir = os.path.join(base_dir, "ligands_pdbqt")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"[ERROR] Không tìm thấy file CSV tại: {csv_path}")
        return

    # Đọc file CSV
    df = pd.read_csv(csv_path, encoding="latin-1")
    
    # Kiểm tra tính hợp lệ của các cột cần thiết
    required_cols = ["canonical_smiles", "compound_name"]
    for col in required_cols:
        if col not in df.columns:
            print(f"[ERROR] File CSV thiếu cột bắt buộc: '{col}'!")
            print(f"Các cột hiện tại trong file của bạn là: {list(df.columns)}")
            return

    print(f"[INFO] Bắt đầu chuẩn bị {len(df)} ligand từ CSV...")
    print("-" * 60)

    success_count = 0
    for i, row in df.iterrows():
        smiles = row["canonical_smiles"]
        raw_name = row["compound_name"]
        ligand_id = row["LigandID"]
        
        # Bỏ qua nếu SMILES bị trống (NaN)
        if pd.isna(smiles) or str(smiles).strip() == "":
            print(f"[SKIP] Bỏ qua dòng {i+1} do cột SMILES bị trống.")
            continue

        # Định dạng tên file sạch: "TênChất_ID.pdbqt" hoặc "TênChất.pdbqt"
        # Thêm LigandID phía sau để tránh trường hợp trùng tên compound_name trong file
        clean_name = sanitize_filename(raw_name)
        if pd.notna(ligand_id):
            clean_id = sanitize_filename(ligand_id)
            file_name = f"{clean_name}_{clean_id}"
        else:
            file_name = clean_name
            
        pdbqt_file = os.path.join(output_dir, f"{file_name}.pdbqt")

        # Tiến hành chuyển đổi
        if convert_smiles_to_pdbqt(smiles, pdbqt_file, raw_name):
            print(f"[SUCCESS] {raw_name} -> {file_name}.pdbqt")
            success_count += 1

    print("-" * 60)
    print(f"[COMPLETED] Đã chuẩn bị thành công {success_count}/{len(df)} ligands.")
    print(f"[INFO] Kết quả lưu tại thư mục: {output_dir}")

if __name__ == "__main__":
    main()