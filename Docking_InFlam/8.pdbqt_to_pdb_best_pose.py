#!/usr/bin/env python3
import os
from pathlib import Path
from openbabel import openbabel

# ==== Cấu hình ====
INPUT_DIR = Path("Best_pose_pdbqt")
OUTPUT_DIR = Path("Best_pose_pdb")
OUTPUT_DIR.mkdir(exist_ok=True)

def extract_first_pose_str(in_file: Path) -> str:
    """Tách MODEL 1 và trả về dưới dạng chuỗi (String), không tạo file tạm."""
    lines = []
    recording = False
    with open(in_file, "r") as f:
        for line in f:
            if line.startswith("MODEL 1"):
                recording = True
            if recording:
                lines.append(line)
            if line.startswith("ENDMDL") and recording:
                break
    return "".join(lines) if lines else ""

def convert_pdbqt_to_pdb(in_file: Path, out_file: Path):
    """Convert pose 1 từ pdbqt sang pdb bằng Open Babel trực tiếp trong bộ nhớ."""
    pose1_content = extract_first_pose_str(in_file)
    
    if not pose1_content:
        print(f"[FAIL] Không tách được pose 1 từ {in_file.name}")
        return

    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdbqt", "pdb")

    mol = openbabel.OBMol()
    
    # Đọc trực tiếp từ chuỗi ký tự thay vì file
    if obConversion.ReadString(mol, pose1_content):
        obConversion.WriteFile(mol, str(out_file))
        print(f"[OK] {in_file.name} (pose 1) → {out_file.name}")
    else:
        print(f"[FAIL] Không đọc được dữ liệu của {in_file.name}")

def main():
    pdbqt_files = sorted(INPUT_DIR.glob("*.pdbqt"))
    if not pdbqt_files:
        print(f"Không tìm thấy file .pdbqt trong {INPUT_DIR}")
        return

    for f in pdbqt_files:
        out_f = OUTPUT_DIR / f.with_suffix(".pdb").name
        convert_pdbqt_to_pdb(f, out_f)

if __name__ == "__main__":
    main()