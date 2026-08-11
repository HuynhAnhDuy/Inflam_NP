import subprocess
from pathlib import Path

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
protein_pdb = Path("mPGES1_5TL9_clean.pdb")
protein_pdbqt = Path("mPGES1_5TL9_clean.pdbqt")

def prepare_receptor_obabel(input_pdb, output_pdbqt):
    print(f"[*] Đang chuẩn bị protein receptor từ: {input_pdb}...")
    
    # Lệnh Open Babel cho receptor:
    # -xr: Giữ nguyên tọa độ gốc
    # -p 7.4: Thêm hydro ở trạng thái pH sinh lý chuẩn
    cmd = [
        "obabel", str(input_pdb), 
        "-O", str(output_pdbqt), 
        "-xr", 
        "-p", "7.4"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"[SUCCESS] Đã tạo thành công file receptor tại: {output_pdbqt}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Lỗi khi chạy Open Babel: {e.stderr.strip()}")
        return False
    except FileNotFoundError:
        print("[ERROR] Không tìm thấy lệnh 'obabel'. Hãy chắc chắn bạn đã cài đặt Open Babel và thêm vào biến môi trường (PATH).")
        return False

if __name__ == "__main__":
    if protein_pdb.exists():
        prepare_receptor_obabel(protein_pdb, protein_pdbqt)
    else:
        print(f"[ERROR] Không tìm thấy file protein: {protein_pdb}")