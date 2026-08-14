import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
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
    ("xgb", "estate"),
    ("xgb", "phychem"),
    ("bilstm", "rdkit"),
    ("bilstm", "ecfp"),
    ("bilstm", "maccs"),
    ("bilstm", "estate"),
    ("bilstm", "phychem"),
]

# === 2. Hàm tính toán metrics ===
def evaluate_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = auc(fpr, tpr)

    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
    auprc = auc(rec_arr, prec_arr)

    mcc = matthews_corrcoef(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob)
    
    # Tính Uncertainty Estimation (Độ bất định: càng thấp càng tốt)
    certainty = np.mean(2 * np.abs(y_prob - 0.5))
    uncertainty = 1.0 - certainty

    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "MCC": mcc,
        "Brier_Score": brier,
        "Uncertainty_Estimation": uncertainty,
    }

# === 3. Chạy đánh giá độc lập cho từng run, tổng hợp Mean ± SD và vẽ Calibration Curve ===
metrics_list = ['AUROC', 'AUPRC', 'MCC', 'Brier_Score', 'Uncertainty_Estimation']
summary_results = []

print("--- ĐANG ĐÁNH GIÁ TỪNG RUN ĐỘC LẬP, TỔNG HỢP MEAN ± SD VÀ VẼ CALIBRATION CURVE ---\n")

for algo, ft in model_keys:
    folder = xgb_dir if algo == "xgb" else bilstm_dir
    name = f"{algo.upper()}_{ft.upper()}"
    
    run_metrics = []
    run_dfs = [] # Lưu các DataFrame của từng run để tính trung bình vẽ biểu đồ
    n_samples_val = 0
    
    # Duyệt qua 3 runs
    for run_idx in [1, 2, 3]:
        path = os.path.join(folder, f"InFlam_full_test_prob_{ft}_run{run_idx}.csv")
        if not os.path.exists(path):
            print(f"[{name}] Thiếu file run {run_idx}. Bỏ qua mô hình này.")
            run_metrics = []
            break
            
        df_run = pd.read_csv(path).dropna(subset=["y_true", "y_prob"])
        n_samples_val = len(df_run)
        
        # Tính metrics cho riêng run này
        res = evaluate_metrics(df_run["y_true"].values, df_run["y_prob"].values)
        run_metrics.append(res)
        run_dfs.append(df_run)
        
    # Nếu thu đủ 3 runs, tiến hành tính Mean & SD và vẽ hình
    if len(run_metrics) == 3:
        df_runs_eval = pd.DataFrame(run_metrics)
        
        row_summary = {
            "Model_Type": "Single Model",
            "Framework": name,
            "N_Samples": n_samples_val
        }
        
        for m in metrics_list:
            mean_val = df_runs_eval[m].mean()
            std_val = df_runs_eval[m].std()
            # Định dạng hiển thị "Mean ± SD" với 3 chữ số thập phân cho file CSV
            row_summary[m] = f"{mean_val:.3f} ± {std_val:.3f}"
            
        summary_results.append(row_summary)
        
        # --- LẤY GIÁ TRỊ MEAN ± SD CỦA BRIER SCORE ĐỂ ĐƯA LÊN BIỂU ĐỒ ---
        brier_mean = df_runs_eval['Brier_Score'].mean()
        brier_std = df_runs_eval['Brier_Score'].std()
        brier_text_sd = f"{brier_mean:.3f} ± {brier_std:.3f}"
        
        # --- TÍNH TOÁN ĐƯỜNG CONG DỰA TRÊN XÁC SUẤT TRUNG BÌNH 3 RUNS ---
        df_mean_curve = run_dfs[0].copy()
        df_mean_curve['y_prob'] = (run_dfs[0]['y_prob'].values + run_dfs[1]['y_prob'].values + run_dfs[2]['y_prob'].values) / 3.0
        
        y_true_curve = df_mean_curve['y_true'].values
        y_prob_curve = df_mean_curve['y_prob'].values
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true_curve, y_prob_curve, n_bins=10, strategy='uniform'
        )
        
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", alpha=0.7)
        
        # Hiển thị Brier dạng Mean ± SD lên Legend của biểu đồ
        label_text = f"{name} (Brier = {brier_text_sd})"
        plt.plot(mean_predicted_value, fraction_of_positives, marker='o', linewidth=2, color='b', label=label_text)
        
        plt.xlabel("Mean Predicted Probability", fontsize=11, fontweight='bold')
        plt.ylabel("Fraction of Positives", fontsize=11, fontweight='bold')
        plt.title(f"Calibration Curve - {name}", fontsize=12, fontweight='bold')
        plt.legend(loc="lower right", fontsize=9, frameon=True)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        
        plt.tight_layout()
        
        svg_filename = f"calibration_single_{name}.svg"
        plt.savefig(svg_filename, format='svg', dpi=300)
        plt.close()
        
        print(f"[{name}] Đã xử lý xong 3 runs, xuất bảng và lưu file hình: {svg_filename}")

# === 4. Xuất kết quả ra file CSV ===
if len(summary_results) > 0:
    df_final = pd.DataFrame(summary_results)
    
    cols = ["Model_Type", "Framework", "N_Samples", "AUROC", "AUPRC", "MCC", "Brier_Score", "Uncertainty_Estimation"]
    df_final = df_final[[c for c in cols if c in df_final.columns]]

    output_csv = "single_models_mean_sd_evaluation.csv"
    df_final.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    print(f"\n=== THÀNH CÔNG! Đã lưu báo cáo CSV tại: {output_csv} ===")
    print(df_final.to_string(index=False))
else:
    print("\n[CẢNH BÁO]: Không có mô hình nào đủ dữ liệu 3 runs để tổng hợp.")