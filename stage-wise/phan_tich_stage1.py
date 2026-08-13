# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# ========= CONFIG =========
INPUT_CSV = "test_day_du_with_predictions.csv"
OUTPUT_CSV = "stage_wise_with_all_steps_SA4.csv"
OUTPUT_DOCKING_CSV = "stage_wise_compounds_for_docking_SA4.csv"  # File riêng chứa danh sách mẫu đem đi Docking

# Các ngưỡng cấu hình cho từng bước lọc
PROB_THRESHOLD = 0.5        # Ngưỡng xác suất cho consensus prediction
MAX_SA_SCORE = 4.0          # Ngưỡng SA score tối đa được chấp nhận (<= 2.0)
DOCKING_THRESHOLD = -7.0    # Ngưỡng điểm docking tốt (<= -7.0)
# ==========================

def main():
    print(f"Đang đọc dữ liệu từ {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    # Đổi tên cột 'True label' thành 'label' nếu cần để khớp với các logic trước đó
    if 'True label' in df.columns:
        df['label'] = df['True label']

    # 1. Bước ML / Consensus Prediction:
    prob_cols = [
        "XGB_ECFP_prob", "XGB_MACCS_prob", "XGB_RDKIT_prob",
        "BiLSTM_ECFP_prob", "BiLSTM_MACCS_prob", "BiLSTM_RDKIT_prob"
    ]
    
    print("Đang đánh giá mô hình đồng thuận (Consensus)...")
    consensus_pass = (df[prob_cols] > PROB_THRESHOLD).all(axis=1)
    df['pass_ml'] = consensus_pass.astype(int)

    # 2. Bước Độc tính (Skin Toxicity & Carcinogenicity):
    if 'skin_toxicity_note' in df.columns and 'carcinogenicity_note' in df.columns:
        pass_skin = df['skin_toxicity_note'].astype(str).str.contains("Non|Negative|0", case=False, na=True)
        pass_carcin = df['carcinogenicity_note'].astype(str).str.contains("Non|Negative|0", case=False, na=True)
        df['pass_toxicity'] = (pass_skin & pass_carcin).astype(int)
    else:
        df['pass_toxicity'] = 1  # Fallback nếu không có cột độc tính

    # 3. Bước SA Score Filtering:
    if 'sa_score' in df.columns:
        df['pass_sa'] = (df['sa_score'] <= MAX_SA_SCORE).astype(int)
    else:
        df['pass_sa'] = 1

    # 4. Xử lý cột pass_docking:
    passed_previous_steps = (df['pass_ml'] == 1) & (df['pass_toxicity'] == 1) & (df['pass_sa'] == 1)

    # Khởi tạo pass_docking bằng 0 cho tất cả
    df['pass_docking'] = 0

    if 'docking_score' in df.columns:
        df.loc[passed_previous_steps & (df['docking_score'] <= DOCKING_THRESHOLD), 'pass_docking'] = 1
    else:
        print("Lưu ý: Không tìm thấy cột 'docking_score' trong file, cột 'pass_docking' sẽ tạm thời là 0 cho tất cả các mẫu.")

    # 5. Lưu thành file CSV hoàn chỉnh tổng thể
    df.to_csv(OUTPUT_CSV, index=False)
    
    # 6. TRÍCH XUẤT VÀ LƯU DANH SÁCH MẪU ĐƯA ĐI DOCKING
    df_for_docking = df[passed_previous_steps].copy()
    df_for_docking.to_csv(OUTPUT_DOCKING_CSV, index=False)

    print(f"\n--- KẾT QUẢ THỐNG KÊ PHỄU ---")
    print(f"Tổng số mẫu: {len(df)}")
    print(f"Số mẫu vượt qua ML Consensus: {df['pass_ml'].sum()}")
    print(f"Số mẫu vượt qua Toxicity: {df['pass_toxicity'].sum()}")
    print(f"Số mẫu vượt qua SA Score (<= {MAX_SA_SCORE}): {df['pass_sa'].sum()}")
    print(f"Số mẫu đủ điều kiện đưa vào Docking: {len(df_for_docking)}")
    
    print(f"\nĐã lưu:")
    print(f"1. File tổng hợp tất cả các bước: {OUTPUT_CSV}")
    print(f"2. File danh sách riêng để chạy Docking: {OUTPUT_DOCKING_CSV}")

if __name__ == "__main__":
    main()