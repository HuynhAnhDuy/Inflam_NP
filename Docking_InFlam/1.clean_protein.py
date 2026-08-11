#!/usr/bin/env python3
import os
import sys

try:
    import pymol
    # -c: chạy không giao diện (headless)
    # -q: chạy trong im lặng (quiet mode)
    pymol.pymol_argv = ['pymol', '-cq'] 
    pymol.finish_launching()
    from pymol import cmd
except ImportError:
    print("[ERROR] Không tìm thấy thư viện PyMOL trong Python.")
    print("--> Hãy cài đặt bằng lệnh: conda install -c schrodinger pymol-open-source")
    sys.exit(1)

# =====================================================================
# BẢN ĐỒ QUY LUẬT LỌC CHO TỪNG PROTEIN (Cực kỳ an toàn và chính xác)
# Key: Ký tự nhận diện trong tên file (không phân biệt chữ hoa/thường)
# Value: Lệnh PyMOL lọc sạch rác nhưng giữ lại cofactor/ion cần thiết
# =====================================================================
FILTERS_MAP = {
    "5TL9": "not polymer.protein and not resn GSH", # Giữ lại protein VÀ phân tử Glutathione (GSH)
}

def clean_pdb_with_pymol(input_path, output_path, file_name):
    """
    Tự động phân tích tên file để áp dụng đúng bộ lọc hóa tin học,
    loại bỏ các phân tử rác nhưng giữ lại các ion/cofactor thiết yếu.
    """
    # Chuyển tên file về chữ thường để so khớp chính xác
    fn_lower = file_name.lower()
    
    # Mặc định nếu không trùng mã nào thì dọn sạch hoàn toàn (not polymer.protein)
    remove_query = "not polymer.protein"
    protein_type = "Unknown (Default: Pure Protein)"

    # Tìm quy luật lọc phù hợp từ bản đồ FILTERS_MAP
    for key, query in FILTERS_MAP.items():
        if key in fn_lower:
            remove_query = query
            protein_type = key.upper()
            break

    try:
        # 1. Load file protein gốc
        cmd.load(input_path, "target_protein")
        
        # 2. Tiến hành xóa nước trước để chắc chắn sạch sẽ
        cmd.remove("solvent")
        
        # 3. Áp dụng query xóa rác đặc thù cho từng protein
        cmd.remove(remove_query)
        
        # 4. Lưu cấu trúc sạch
        cmd.save(output_path, "target_protein")
        
        print(f"    ==> [MATCHED] Nhận diện protein: {protein_type} | Lệnh dùng: remove {remove_query}")
        
        # 5. Giải phóng vùng nhớ
        cmd.delete("all")
        return True
    except Exception as e:
        print(f"[ERROR] Lỗi khi xử lý file {input_path} qua PyMOL: {e}")
        cmd.delete("all")
        return False

def main():
    # Thư mục chứa các file PDB thô ban đầu (ví dụ: 5KIR.pdb, 4NOS_raw.pdb, v.v.)
    raw_dir = "Protein_original" 
    # Thư mục sẽ chứa các file PDB đã dọn sạch chuẩn hóa
    clean_dir = "Protein_clean" 

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)

    pdb_files = [f for f in os.listdir(raw_dir) if f.endswith(".pdb")]

    if not pdb_files:
        print(f"[WARNING] Thư mục '{raw_dir}' đang trống!")
        print(f"--> Hãy bỏ các file PDB gốc vào thư mục '{raw_dir}' rồi chạy lại script.")
        return

    print(f"[INFO] Bắt đầu dọn dẹp {len(pdb_files)} file PDB bằng PyMOL thông minh...")
    print("-" * 80)

    for pdb_file in pdb_files:
        input_path = os.path.join(raw_dir, pdb_file)
        
        base_name = os.path.splitext(pdb_file)[0]
        # Thống nhất tên đầu ra kết thúc bằng '_clean.pdb' để đồng bộ với code chuyển .pdbqt tiếp theo
        output_path = os.path.join(clean_dir, f"{base_name}_clean.pdb")

        print(f"[PROCESSING] Đang xử lý file: {pdb_file}")
        if clean_pdb_with_pymol(input_path, output_path, pdb_file):
            print(f"    ==> [SUCCESS] Đã lưu protein chuẩn tại: {output_path}\n")
        else:
            print(f"    ==> [FAILED] Thất bại khi lọc file: {pdb_file}\n")

    print("-" * 80)
    print(f"[INFO] Hoàn thành! Toàn bộ {len(pdb_files)} receptor sạch đã được lưu gọn gàng tại '{clean_dir}'.")

if __name__ == "__main__":
    main()