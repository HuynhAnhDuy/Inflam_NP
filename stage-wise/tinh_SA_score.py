# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
import sascorer  # Đảm bảo file sascorer.py nằm trong thư mục làm việc

# ========= CONFIG =========
INPUT_CSV       = "test_day_du.csv"
OUTPUT_CSV      = "test_day_du_with_sa.csv" # File đầu ra chứa giá trị SA score
# ==========================

def calc_sa_score(mol):
    try:
        return sascorer.calculateScore(mol) if mol else np.nan
    except Exception:
        return np.nan

def main():
    print(f"Đang đọc dữ liệu từ {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    # 1. Parsing SMILES
    tqdm.pandas(desc="Parsing SMILES")
    df['mol'] = df['canonical_smiles'].progress_apply(lambda s: Chem.MolFromSmiles(str(s)) if pd.notna(s) else None)

    # 2. Tính toán SA Score cho tất cả các mẫu (Không lọc)
    print("\nĐang tính toán SA Score cho toàn bộ các mẫu...")
    tqdm.pandas(desc="Calculating SA Score")
    df['sa_score'] = df['mol'].progress_apply(calc_sa_score)

    # Xóa cột tạm 'mol' (RDKit Mol object) trước khi lưu file vì không serialize được ra CSV
    if 'mol' in df.columns:
        df = df.drop(columns=['mol'])

    # --- CHẨN ĐOÁN DỮ LIỆU SA SCORE ---
    print("\n--- PHÂN TÍCH DỮ LIỆU SA SCORE ---")
    valid_sa = df['sa_score'].dropna()
    print(f"Tổng số mẫu trong file: {len(df)}")
    print(f"Số mẫu tính được SA score hợp lệ: {len(valid_sa)}")

    if len(valid_sa) > 0:
        print(f"SA Score thấp nhất: {valid_sa.min():.2f}")
        print(f"SA Score cao nhất: {valid_sa.max():.2f}")
    else:
        print("CẢNH BÁO: Không có mẫu nào tính được SA Score hợp lệ!")

    n_nan = df['sa_score'].isna().sum()
    if n_nan > 0:
        print(f"Số mẫu lỗi (NaN/Không thể tính SA): {n_nan}")
        print("Mẫu lỗi đầu tiên (SMILES):", df[df['sa_score'].isna()]['canonical_smiles'].head(5).values)
    # ---------------------------------

    # 3. Lưu toàn bộ dữ liệu kèm cột sa_score mới vào file output
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\nHoàn thành! Đã lưu kết quả vào: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()