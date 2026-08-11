#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path
from rdkit import Chem

# =====================================================================
#                         1. KHỐI CẤU HÌNH & TỪ ĐIỂN
# =====================================================================
RAW_PDB_DIR = Path(r"D:\Andy\Inflam_NP\Docking_InFlam\Protein_original")
BASE_OUTPUT_DIR = Path(r"D:\Andy\Inflam_NP\Docking_InFlam\Processed_CoCrystal")
BASE_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

PROTEIN_LIGAND_MAP = {
    "6NCF": "AF7",    # 5LOX
    "1EQH": "FLP",    # COX-1
    "3LN1": "CEL",    # COX-2
    "5IKR": "ID8",    # COX-2
    "5TL9": "7DN",    # mPGES1
}

# =====================================================================
#                         2. HÀM XỬ LÝ LIGAND ĐỒNG KẾT TINH
# =====================================================================
def extract_and_prepare_ligand(complex_pdb_path, resname, output_pdbqt_path):
    ligand_pdb_lines = []
    with open(complex_pdb_path, "r", encoding="utf-8") as f:
        for line in f:
            # Lọc cả HETATM và các dòng CONECT liên quan nếu có, hoặc đơn giản là lọc đúng RESNAME
            if line.startswith("HETATM") and resname in line:
                ligand_pdb_lines.append(line)
                
    if not ligand_pdb_lines:
        print(f"      [ERROR] Không tìm thấy RESNAME '{resname}' trong file PDB!")
        return False
        
    temp_lig_pdb = BASE_OUTPUT_DIR / "temp_ligand.pdb"
    with open(temp_lig_pdb, "w", encoding="utf-8") as f:
        f.writelines(ligand_pdb_lines)
        f.write("END\n")

    try:
        # Sử dụng trực tiếp Open Babel để chuyển đổi từ PDB sang PDBQT của co-ligand
        # Thêm cờ -p để giữ nguyên protonation state (trạng thái ion hóa) sinh lý nếu có, 
        # hoặc dùng -xr để giữ nguyên tọa độ gốc tuyệt đối không dịch chuyển.
        cmd = [
            "obabel", str(temp_lig_pdb), 
            "-O", str(output_pdbqt_path), 
            "-h",            # Thêm hydro ở trạng thái pH sinh lý (7.4)
            "--partialcharge", "gasteiger" # Gán điện tích Gasteiger bắt buộc cho Vina
        ]
        
        # Chạy lệnh và bắt lỗi chi tiết nếu Open Babel thất bại
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if temp_lig_pdb.exists(): 
            temp_lig_pdb.unlink()
            
        if result.returncode != 0:
            print(f"      [ERROR] Open Babel thất bại: {result.stderr.strip()}")
            return False
            
        return True
            
    except Exception as e:
        print(f"      [ERROR] Lỗi xử lý ligand: {e}")
        if temp_lig_pdb.exists(): 
            temp_lig_pdb.unlink()
        return False

# =====================================================================
#                         3. VÒNG LẶP CHẠY HÀNG LOẠT
# =====================================================================
def main():
    print("=" * 70)
    print(" XỬ LÝ TÁCH & CHUẨN HÓA LIGAND ĐỒNG KẾT TINH (REFERENCE) ")
    print("=" * 70)
    
    for pdb_id, lig_resname in PROTEIN_LIGAND_MAP.items():
        print(f"\n[PROCESSING] Protein ID: {pdb_id} | Ligand: {lig_resname}")
        
        complex_pdb_path = None
        for file_path in RAW_PDB_DIR.glob("*.pdb"):
            if pdb_id.lower() in file_path.name.lower():
                complex_pdb_path = file_path
                break
                
        if not complex_pdb_path:
            print(f"   [SKIP] Không tìm thấy file PDB chứa mã '{pdb_id}' trong {RAW_PDB_DIR}")
            continue
            
        print(f"   -> Tìm thấy file: {complex_pdb_path.name}")
        
        ligand_out = BASE_OUTPUT_DIR / f"{lig_resname}_ref.pdbqt"
        
        print(f"   -> Đang tách và chuẩn hóa ligand đồng kết tinh...")
        lig_success = extract_and_prepare_ligand(complex_pdb_path, lig_resname, ligand_out)
        
        if lig_success:
            print(f"   [SUCCESS] Hoàn tất! Đã lưu file tại: {ligand_out}")
        else:
            print(f"   [WARNING] Hệ thống {pdb_id} xử lý ligand có lỗi phát sinh.")

    print("\n" + "=" * 70)
    print("[COMPLETED] Đã hoàn thành toàn bộ danh sách ligand đối chứng!")
    print("=" * 70)

if __name__ == "__main__":
    main()