# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from tqdm import tqdm
import sascorer  # Đảm bảo file sascorer.py nằm trong thư mục làm việc

# ========= CONFIG =========
INPUT_CSV       = "test_final_merged.csv"
OUTPUT_X_CSV    = "x.csv"
OUTPUT_Y_CSV    = "y.csv"
OUTPUT_REPORT   = "screening_sa_analysis.csv" # Báo cáo chi tiết các ngưỡng SA

MAX_LIPINSKI_VIOL = 1
# ==========================

def calc_ro5_violations(mol):
    if not mol: return np.nan
    mw   = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd  = rdMolDescriptors.CalcNumHBD(mol)
    hba  = rdMolDescriptors.CalcNumHBA(mol)
    viol = (mw > 500) + (logp > 5) + (hbd > 5) + (hba > 10)
    return viol

def calc_sa_score(mol):
    try:
        return sascorer.calculateScore(mol) if mol else np.nan
    except Exception:
        return np.nan

def main():
    df = pd.read_csv(INPUT_CSV)
    
    # 1. Parsing SMILES
    tqdm.pandas(desc="Parsing SMILES")
    df['mol'] = df['canonical_smiles'].progress_apply(lambda s: Chem.MolFromSmiles(str(s)) if pd.notna(s) else None)

    # 2. Bước 1: Sàng lọc RO5
    df['ro5_viol'] = df['mol'].apply(calc_ro5_violations)
    df_ro5 = df[df['ro5_viol'] <= MAX_LIPINSKI_VIOL].copy()
    
    # 3. Bước 2: Tính SA Score cho các mẫu qua RO5
    print("\nĐang tính toán SA Score cho các mẫu...")
    df_ro5['sa_score'] = df_ro5['mol'].progress_apply(calc_sa_score)

    # --- CHẨN ĐOÁN DỮ LIỆU SA SCORE ---
    print("\n--- PHÂN TÍCH DỮ LIỆU SA SCORE ---")
    valid_sa = df_ro5['sa_score'].dropna()
    print(f"Tổng số mẫu sau RO5: {len(df_ro5)}")
    print(f"Số mẫu tính được SA score hợp lệ: {len(valid_sa)}")

    if len(valid_sa) > 0:
        print(f"SA Score thấp nhất: {valid_sa.min():.2f}")
        print(f"SA Score cao nhất: {valid_sa.max():.2f}")
        print(f"Số mẫu có SA score <= 10: {len(df_ro5[df_ro5['sa_score'] <= 10])}")
    else:
        print("CẢNH BÁO: Không có mẫu nào tính được SA Score hợp lệ!")

    n_nan = df_ro5['sa_score'].isna().sum()
    if n_nan > 0:
        print(f"Số mẫu lỗi (NaN/Không thể tính SA): {n_nan}")
        print("Mẫu lỗi đầu tiên (SMILES):", df_ro5[df_ro5['sa_score'].isna()]['canonical_smiles'].head(5).values)
    # ---------------------------------

    # --- 4. Phân tích với các ngưỡng SA từ 1 đến 10 (Đã sửa lại thụt lề chuẩn) ---
    report_data = []
    
    print("\n=== ĐANG PHÂN TÍCH CÁC NGƯỠNG SA (1-10) ===")
    for sa_threshold in range(1, 11):
        df_kept = df_ro5[df_ro5['sa_score'] <= sa_threshold]
        df_dropped = df_ro5[df_ro5['sa_score'] > sa_threshold]
        
        report_data.append({
            "SA_Threshold": sa_threshold,
            "Kept_Total": len(df_kept),
            "Kept_Positive": int((df_kept['Label'] == 1).sum()),
            "Kept_Negative": int((df_kept['Label'] == 0).sum()),
            "Dropped_Total": len(df_dropped),
            "Dropped_Positive": int((df_dropped['Label'] == 1).sum()),
            "Dropped_Negative": int((df_dropped['Label'] == 0).sum())
        })
        print(f"Ngưỡng {sa_threshold}: Giữ {len(df_kept)} mẫu.")

    # Lưu báo cáo
    df_report = pd.DataFrame(report_data)
    df_report.to_csv(OUTPUT_REPORT, index=False)
    
    # Xuất file mặc định với ngưỡng SA = 2.0 (hoặc fallback nếu bằng 0)
    DEFAULT_SA = 2.0
    df_final = df_ro5[df_ro5['sa_score'] <= DEFAULT_SA].copy()
    if len(df_final) == 0: 
        print(f"\n[CẢNH BÁO] Ngưỡng SA mặc định ({DEFAULT_SA}) giữ lại 0 mẫu. Đang áp dụng fallback giữ lại toàn bộ tập RO5.")
        df_final = df_ro5
    
    df_final[['Index', 'canonical_smiles']].to_csv(OUTPUT_X_CSV, index=False)
    df_final[['Index', 'Label']].to_csv(OUTPUT_Y_CSV, index=False)

    print(f"\nHoàn thành! Đã lưu:")
    print(f"- {OUTPUT_REPORT} (File CSV chứa thống kê 10 ngưỡng SA)")
    print(f"- {OUTPUT_X_CSV} và {OUTPUT_Y_CSV} (Sử dụng ngưỡng SA={DEFAULT_SA})")

if __name__ == "__main__":
    main()