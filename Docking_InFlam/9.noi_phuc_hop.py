#!/usr/bin/env python3
from pathlib import Path

# ==== 1. CẤU HÌNH THƯ MỤC ====
# Hãy đổi đường dẫn dưới đây theo đúng tên thư mục thực tế của bạn
PROTEIN_DIR = Path("Protein_clean")  # Thư mục chứa các file *_clean.pdb
LIGAND_DIR = Path("Best_pose_pdb")    # Thư mục chứa các file ligand pose 1 (.pdb)
OUTPUT_DIR = Path("Complex_PDB")      # Thư mục lưu kết quả phức hợp
OUTPUT_DIR.mkdir(exist_ok=True)

def find_matching_protein(protein_prefix: str) -> Path | None:
    """
    Tìm file protein trong PROTEIN_DIR dựa vào prefix.
    Ví dụ: prefix 'COX2' sẽ tìm thấy 'COX2-5kir_clean.pdb'
    """
    # Tìm file bắt đầu bằng prefix (ví dụ: COX2-*)
    matches = list(PROTEIN_DIR.glob(f"{protein_prefix}-*.pdb"))
    
    if not matches:
        # Nếu không tìm thấy dạng COX2-*, thử tìm file có chứa prefix
        matches = [f for f in PROTEIN_DIR.glob("*.pdb") if f.name.startswith(protein_prefix)]
        
    if matches:
        return matches[0]  # Trả về file khớp đầu tiên tìm được
    return None

def merge_protein_and_ligand(protein_file: Path, ligand_file: Path, output_file: Path):
    """Đọc và ghép nội dung Protein + Ligand thành file Complex .pdb hoàn chỉnh."""
    complex_lines = []

    # 1. Đọc file Protein (lấy ATOM, HETATM, TER)
    with open(protein_file, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM", "TER")):
                complex_lines.append(line)
    
    # Đảm bảo phân cách chuỗi rõ ràng
    if not complex_lines[-1].startswith("TER"):
        complex_lines.append("TER\n")

    # 2. Đọc file Ligand (lấy ATOM, HETATM)
    with open(ligand_file, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                complex_lines.append(line)
                
    complex_lines.append("END\n")

    # 3. Ghi file Phức hợp
    with open(output_file, "w") as f:
        f.writelines(complex_lines)

import re
from pathlib import Path

# ... các phần cấu hình và hàm giữ nguyên ...

def main():
    ligand_files = sorted(LIGAND_DIR.glob("*.pdb"))
    
    if not ligand_files:
        print(f"[ERR] Không thấy file .pdb nào trong {LIGAND_DIR}")
        return

    count = 0
    for lig_path in ligand_files:
        lig_filename = lig_path.name
        
        # Tách lấy chữ đầu tiên trước dấu '_' hoặc '-' 
        # (Ví dụ: 'MMP1-Lutein...' hay 'COX2_Lutein...' đều ra 'MMP1' hoặc 'COX2')
        prefix = re.split(r'[_|-]', lig_filename)[0]

        # Tìm protein khớp với prefix
        prot_path = find_matching_protein(prefix)

        if prot_path and prot_path.exists():
            out_path = OUTPUT_DIR / f"Complex_{lig_path.name}"
            merge_protein_and_ligand(prot_path, lig_path, out_path)
            print(f"[OK] Ghép: {prot_path.name} + {lig_path.name} -> {out_path.name}")
            count += 1
        else:
            print(f"[WARN] Không tìm thấy file Protein tương ứng cho Prefix '{prefix}' (File: {lig_filename})")

    print(f"\n---> Hoàn tất! Đã tạo thành công {count} file phức hợp trong thư mục '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()