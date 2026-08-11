#!/usr/bin/env python3
import os
import sys

try:
    from openbabel import openbabel
except ImportError:
    print("[ERROR] Không tìm thấy thư viện 'openbabel'. Hãy cài đặt bằng: conda install -c conda-forge openbabel")
    sys.exit(1)

# ==========================
# Function helpers
# ==========================
def change_pdb_pdbqt(input_file, output_file):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdb", "pdbqt")
    
    # Tự động thêm Hydro (cực kỳ quan trọng để tính điện tích docking)
    obConversion.AddOption("h", openbabel.OBConversion.GENOPTIONS) 
    
    mol = openbabel.OBMol()
    
    success = obConversion.ReadFile(mol, input_file)
    if not success:
        print(f"[ERROR] Không thể đọc file: {input_file}. Vui lòng kiểm tra lại cấu trúc.")
        return False
        
    obConversion.WriteFile(mol, output_file)
    return True

def clean_pdbqt(input_path, output_path):
    # Đọc và loại bỏ các dòng ROOT, BRANCH... để tạo receptor cứng (rigid)
    with open(input_path, 'r') as infile:
        lines = infile.readlines()
        
    with open(output_path, 'w') as outfile:
        for line in lines:
            if not line.startswith(('ROOT','ENDROOT','BRANCH','ENDBRANCH','TORSDOF')):
                outfile.write(line)
                
    # Xóa file trung gian tạm thời để tránh rác
    if os.path.exists(input_path) and input_path != output_path:
        os.remove(input_path)

# ==========================
# Main workflow
# ==========================
def main():
    input_dir = "Protein_clean"     # Thư mục chứa các file .pdb đầu vào của bạn
    output_dir = "Protein_prepared"   # Thư mục MỚI sẽ chứa các file .pdbqt đầu ra
    
    # Tạo thư mục đầu vào nếu chưa có
    if not os.path.exists(input_dir):
        os.makedirs(input_dir, exist_ok=True)
        print(f"[WARNING] Đã tạo thư mục đầu vào '{input_dir}'.")
        print(f"--> Hãy copy các file .pdb cần chuyển đổi vào thư mục '{input_dir}' rồi chạy lại code.")
        return

    # Tạo thư mục đầu ra
    os.makedirs(output_dir, exist_ok=True)

    # Tìm tất cả các file .pdb trong thư mục đầu vào (chấp nhận cả đuôi .pdb thường và _clean.pdb)
    pdb_files = [f for f in os.listdir(input_dir) if f.endswith(".pdb")]
    
    if not pdb_files:
        print(f"[ERROR] Không tìm thấy file .pdb nào trong thư mục '{input_dir}'.")
        print(f"--> Hãy chắc chắn rằng bạn đã đặt các file cấu trúc protein (.pdb) vào trong '{input_dir}'.")
        return

    print(f"[INFO] Tìm thấy {len(pdb_files)} file PDB cần xử lý. Bắt đầu chuyển đổi...")
    print("-" * 50)

    for pdb_file in pdb_files:
        # Tách tên file gốc bỏ đuôi .pdb (và bỏ luôn đuôi _clean nếu có để tên file output gọn gàng)
        base_name = pdb_file.replace(".pdb", "").replace("_clean", "")
        
        # Đường dẫn file
        pdb_path = os.path.join(input_dir, pdb_file)
        temp_pdbqt_path = os.path.join(output_dir, f"{base_name}_temp.pdbqt")
        final_pdbqt_clean = os.path.join(output_dir, f"{base_name}_clean.pdbqt")

        # Tiến hành chuyển đổi và làm sạch
        if change_pdb_pdbqt(pdb_path, temp_pdbqt_path):
            clean_pdbqt(temp_pdbqt_path, final_pdbqt_clean)
            print(f"[SUCCESS] Đã xử lý xong: {pdb_file} -> {output_dir}/{base_name}_clean.pdbqt")
        else:
            print(f"[FAILED] Thất bại khi xử lý file: {pdb_file}")

    print("-" * 50)
    print(f"[INFO] Hoàn thành! Tất cả kết quả sạch đã được lưu tại thư mục '{output_dir}'.")

if __name__ == "__main__":
    main()