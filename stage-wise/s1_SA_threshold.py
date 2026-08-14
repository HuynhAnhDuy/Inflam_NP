# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# ========= CONFIG =========
INPUT_CSV = "test_day_du_with_predictions.csv"
OUTPUT_DATA_CSV = "full_processed_data.csv"
STATS_CSV = "statistics_summary_for_docking.csv"

PROB_THRESHOLD = 0.5
# ==========================

def main():
    print(f"Đang đọc dữ liệu từ {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    total_initial = len(df)
    
    # 1. Consensus Prediction (ML) -> Tạo cột 'Predicted label' và 'pass_ml' (dạng yes/no)
    prob_cols = [
        "XGB_ECFP_prob", "XGB_MACCS_prob", "XGB_RDKIT_prob",
        "BiLSTM_ECFP_prob", "BiLSTM_MACCS_prob", "BiLSTM_RDKIT_prob"
    ]
    existing_prob_cols = [c for c in prob_cols if c in df.columns]
    
    if existing_prob_cols:
        pass_ml_bool = (df[existing_prob_cols] > PROB_THRESHOLD).all(axis=1)
        df['pass_ml'] = np.where(pass_ml_bool, 'yes', 'no')
        # Predicted label vẫn dùng số (1 hoặc 0) vì đây là nhãn dự đoán để tính TP, FP, TN, FN
        df['Predicted label'] = pass_ml_bool.astype(int)
    else:
        df['pass_ml'] = 'yes'
        df['Predicted label'] = 1
    
    # 2. Toxicity Filtering (pass_toxicity: 'yes' nếu cả 2 cột note bằng 0, ngược lại 'no')
    if 'skin_toxicity_note' in df.columns and 'carcinogenicity_note' in df.columns:
        pass_skin = (pd.to_numeric(df['skin_toxicity_note'], errors='coerce') == 0)
        pass_carcin = (pd.to_numeric(df['carcinogenicity_note'], errors='coerce') == 0)
        df['pass_toxicity'] = np.where(pass_skin & pass_carcin, 'yes', 'no')
    else:
        df['pass_toxicity'] = 'yes'

    stats_list = []
    
    # 3. Lặp qua SA threshold từ 2 đến 10 để chuẩn bị danh sách đi Docking
    for sa_threshold in range(2, 11):
        if 'sa_score' in df.columns:
            pass_sa_bool = (df['sa_score'] <= sa_threshold)
            df[f'pass_sa_{sa_threshold}'] = np.where(pass_sa_bool, 'yes', 'no')
        else:
            pass_sa_bool = pd.Series([True] * len(df))
            df[f'pass_sa_{sa_threshold}'] = 'yes'
            
        # Điều kiện các bước trước khi đưa đi Docking (ML = 'yes' + Toxicity = 'yes' + SA hiện tại = 'yes')
        passed_for_docking = (df['pass_ml'] == 'yes') & (df['pass_toxicity'] == 'yes') & (df[f'pass_sa_{sa_threshold}'] == 'yes')
        
        # Lưu cột đánh dấu mẫu nào được chọn đi docking ở ngưỡng SA này (dạng yes/no)
        df[f'select_for_docking_sa{sa_threshold}'] = np.where(passed_for_docking, 'yes', 'no')
        
        # Thống kê số lượng và phần trăm giữ lại so với ban đầu
        subset = df[passed_for_docking]
        count = len(subset)
        pct = (count / total_initial) * 100 if total_initial > 0 else 0
        
        # Tính toán TP, FP, TN, FN dựa trên True label và Predicted label
        tp = fp = tn = fn = 0
        if 'Predicted label' in df.columns and 'True label' in df.columns and not subset.empty:
            y_pred = pd.to_numeric(subset['Predicted label'], errors='coerce').fillna(0).astype(int)
            y_true = pd.to_numeric(subset['True label'], errors='coerce').fillna(0).astype(int)
            
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            
        stats = {
            'SA_Threshold': sa_threshold,
            'Count_For_Docking': count,
            'Percentage_For_Docking': pct,
            'TP': tp, 
            'FP': fp, 
            'TN': tn, 
            'FN': fn
        }
        stats_list.append(stats)

    # Lưu file tổng hợp
    df.to_csv(OUTPUT_DATA_CSV, index=False)
    pd.DataFrame(stats_list).to_csv(STATS_CSV, index=False)
    
    print(f"\nHoàn tất xử lý!")
    print(f"- File tổng hợp dữ liệu: {OUTPUT_DATA_CSV}")
    print(f"- File thống kê cho các ngưỡng SA (2-10): {STATS_CSV}")

if __name__ == "__main__":
    main()