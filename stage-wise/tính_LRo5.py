# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from tqdm import tqdm

# ========= CONFIG =========
INPUT_CSV       = "test_goc.csv"
OUTPUT_X_CSV    = "x.csv"
OUTPUT_Y_CSV    = "y.csv"
OUTPUT_REPORT   = "screening_phychem_analysis.csv" # Báo cáo thống kê tổng quan sau lọc hóa lý

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

def main():
    df = pd.read_csv(INPUT_CSV)
    
    # 1. Parsing SMILES
    tqdm.pandas(desc="Parsing SMILES")
    df['mol'] = df['canonical_smiles'].progress_apply(lambda s: Chem.MolFromSmiles(str(s)) if pd.notna(s) else None)

    # 2. Sàng lọc hóa lý (Lipinski's Rule of Five)
    df['ro5_viol'] = df['mol'].apply(calc_ro5_violations)
    
    # Lọc các mẫu đạt tiêu chuẩn hóa lý
    df_phychem_passed = df[df['ro5_viol'] <= MAX_LIPINSKI_VIOL].copy()
    df_phychem_dropped = df[df['ro5_viol'] > MAX_LIPINSKI_VIOL].copy()

    # --- THỐNG KÊ KẾT QUẢ LỌC HÓA LÝ ---
    print("\n--- PHÂN TÍCH KẾT QUẢ LỌC HÓA LÝ (PHYSICOCHEMICAL FILTERING) ---")
    print(f"Tổng số mẫu ban đầu: {len(df)}")
    print(f"Số mẫu đạt (pass) qua lọc hóa lý: {len(df_phychem_passed)}")
    print(f"Số mẫu bị loại (fail): {len(df_phychem_dropped)}")

    # Tạo báo cáo tóm tắt
    report_data = [{
        "Total_Initial": len(df),
        "Passed_Total": len(df_phychem_passed),
        "Passed_Positive": int((df_phychem_passed['Label'] == 1).sum()) if 'Label' in df_phychem_passed.columns else 0,
        "Passed_Negative": int((df_phychem_passed['Label'] == 0).sum()) if 'Label' in df_phychem_passed.columns else 0,
        "Dropped_Total": len(df_phychem_dropped),
        "Dropped_Positive": int((df_phychem_dropped['Label'] == 1).sum()) if 'Label' in df_phychem_dropped.columns else 0,
        "Dropped_Negative": int((df_phychem_dropped['Label'] == 0).sum()) if 'Label' in df_phychem_dropped.columns else 0
    }]

    df_report = pd.DataFrame(report_data)
    df_report.to_csv(OUTPUT_REPORT, index=False)
    
    # --- XUẤT FILE X VÀ Y CHO CÁC MẪU PASS ---
    # Kiểm tra cột Index trước khi xuất, nếu không có thì dùng index mặc định của DataFrame
    x_columns = ['Index', 'canonical_smiles'] if 'Index' in df_phychem_passed.columns else ['canonical_smiles']
    y_columns = ['Index', 'Label'] if 'Index' in df_phychem_passed.columns else ['Label']

    df_phychem_passed[x_columns].to_csv(OUTPUT_X_CSV, index=False)
    df_phychem_passed[y_columns].to_csv(OUTPUT_Y_CSV, index=False)

    print(f"\nHoàn thành! Đã lưu:")
    print(f"- {OUTPUT_REPORT} (Báo cáo tổng kết lọc hóa lý)")
    print(f"- {OUTPUT_X_CSV} và {OUTPUT_Y_CSV} (Chứa các hợp chất pass qua bước lọc hóa lý)")

if __name__ == "__main__":
    main()