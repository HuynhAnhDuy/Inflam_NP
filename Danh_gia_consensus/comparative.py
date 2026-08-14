from itertools import combinations
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    brier_score_loss,
    matthews_corrcoef,
    precision_recall_curve,
    roc_curve,
)

# === 1. Cấu hình đường dẫn ===
xgb_dir = r"D:\Andy\Inflam_NP\Predictive_models\Prob_InFlam_Full\Prob_2026-08-10_xgb"
bilstm_dir = r"D:\Andy\Inflam_NP\Predictive_models\Prob_InFlam_Full\Prob_2026-08-10_BiLSTM"

model_keys = [
    ("xgb", "rdkit"),
    ("xgb", "ecfp"),
    ("xgb", "maccs"),
    ("bilstm", "rdkit"),
    ("bilstm", "ecfp"),
    ("bilstm", "maccs"),
]

# === 2. Tải dữ liệu 3 runs cho từng mô hình đơn ===
print("--- ĐANG TẢI DỮ LIỆU 3 RUNS CHO TỪNG MÔ HÌNH ĐƠN ---")
model_runs_data = {}

for algo, ft in model_keys:
    folder = xgb_dir if algo == "xgb" else bilstm_dir
    name = f"{algo.upper()}_{ft.upper()}"
    
    run_dfs = []
    success = True
    for run_idx in [1, 2, 3]:
        path = os.path.join(folder, f"InFlam_full_test_prob_{ft}_run{run_idx}.csv")
        if not os.path.exists(path):
            success = False
            break
        df_run = pd.read_csv(path).dropna(subset=["y_true", "y_prob"])
        run_dfs.append(df_run)
        
    if success:
        model_runs_data[name] = run_dfs

model_names = list(model_runs_data.keys())

# === 3. Hàm tính toán metrics ===
def evaluate_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = auc(fpr, tpr)

    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
    auprc = auc(rec_arr, prec_arr)

    mcc = matthews_corrcoef(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob)
    certainty = np.mean(2 * np.abs(y_prob - 0.5))

    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "MCC": mcc,
        "Brier_Score": brier,
        "Prediction_Certainty": certainty,
    }

metrics_list = ['AUROC', 'AUPRC', 'MCC', 'Brier_Score', 'Prediction_Certainty']
all_evaluations = []

# === 4. Đánh giá từng Run độc lập cho Single Models ===
for name in model_names:
    run_metrics = []
    n_samples_val = 0
    for run_df in model_runs_data[name]:
        n_samples_val = len(run_df)
        res = evaluate_metrics(run_df["y_true"].values, run_df["y_prob"].values)
        run_metrics.append(res)
        
    df_runs_eval = pd.DataFrame(run_metrics)
    row_summary = {
        "Model_Type": "Single Model",
        "Framework": name,
        "N_Samples": n_samples_val
    }
    
    # Lưu cả giá trị thô (để so sánh) và dạng chuỗi Mean ± SD
    for m in metrics_list:
        mean_val = df_runs_eval[m].mean()
        std_val = df_runs_eval[m].std()
        row_summary[f"{m}_mean"] = mean_val
        row_summary[f"{m}_std"] = std_val
        row_summary[m] = f"{mean_val:.3f} ± {std_val:.3f}"
        
    all_evaluations.append(row_summary)

# === 5. Đánh giá Consensus Models (Combo 2 -> 6) qua từng Run ===
for r in range(2, len(model_names) + 1):
    for combo in combinations(model_names, r):
        combo_run_metrics = []
        n_samples_val = 0
        valid_combo = True
        
        for run_idx in range(3):
            probs_dict = {}
            y_true_ref = None
            
            for m in combo:
                df_run = model_runs_data[m][run_idx]
                probs_dict[m] = df_run["y_prob"].reset_index(drop=True)
                if y_true_ref is None:
                    y_true_ref = df_run["y_true"].reset_index(drop=True)
                    
            df_combo_probs = pd.DataFrame(probs_dict).dropna()
            if len(df_combo_probs) == 0:
                valid_combo = False
                break
                
            valid_indices = df_combo_probs.index
            y_true_combo = y_true_ref.loc[valid_indices].values
            consensus_probs = df_combo_probs.mean(axis=1).values
            n_samples_val = len(y_true_combo)
            
            res = evaluate_metrics(y_true_combo, consensus_probs)
            combo_run_metrics.append(res)
            
        if valid_combo and len(combo_run_metrics) == 3:
            df_combo_runs_eval = pd.DataFrame(combo_run_metrics)
            row_summary = {
                "Model_Type": f"Combo {r}",
                "Framework": " + ".join(combo),
                "N_Samples": n_samples_val
            }
            
            for m in metrics_list:
                mean_val = df_combo_runs_eval[m].mean()
                std_val = df_combo_runs_eval[m].std()
                row_summary[f"{m}_mean"] = mean_val
                row_summary[f"{m}_std"] = std_val
                row_summary[m] = f"{mean_val:.3f} ± {std_val:.3f}"
                
            all_evaluations.append(row_summary)

df_all = pd.DataFrame(all_evaluations)

# === 6. Lọc cấu hình tốt nhất cho từng Model_Type dựa trên AUROC trung bình ===
best_indices = df_all.groupby('Model_Type')['AUROC_mean'].idxmax()
best_configs = df_all.loc[best_indices].copy()

# === 7. Sắp xếp theo Prediction_Certainty giảm dần, Brier_Score tăng dần ===
best_configs = best_configs.sort_values(
    by=['Prediction_Certainty_mean', 'Brier_Score_mean'], 
    ascending=[False, True]
).reset_index(drop=True)

# === 8. Thêm cột Rank và xuất file ===
best_configs.insert(0, 'Rank', range(1, len(best_configs) + 1))

# Giữ lại các cột chính thức cần xuất báo cáo (gồm giá trị dạng Mean ± SD)
cols_to_keep = ['Rank', 'Model_Type', 'Framework', 'N_Samples', 'AUROC', 'MCC', 'AUPRC', 'Brier_Score', 'Prediction_Certainty']
df_final_output = best_configs[[c for c in cols_to_keep if c in best_configs.columns]]

print("\n--- CÁC CẤU HÌNH TỐT NHẤT (ĐÃ CÓ MEAN ± SD CHÍNH XÁC) ---")
print(df_final_output.to_string(index=False))

df_final_output.to_csv("ensemble_best_configs.csv", encoding='utf-8-sig', index=False)
print("\nĐã xuất file ensemble_best_configs.csv thành công với SD thực tế từ 3 runs!")