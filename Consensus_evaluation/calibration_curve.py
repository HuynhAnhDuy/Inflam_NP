import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression
from statsmodels.nonparametric.smoothers_lowess import lowess

# === 1. Cấu hình đường dẫn ===
xgb_dir = r"D:\Andy\Inflam_NP\Predictive_models\Prob_InFlam_Full\Prob_XGB"
bilstm_dir = r"D:\Andy\Inflam_NP\Predictive_models\Prob_InFlam_Full\Prob_BiLSTM"

# Định nghĩa 6 cấu hình bạn yêu cầu
configs_to_plot = {
    "XGB_RDKIT": [("xgb", "rdkit")],
    "Consensus 2": [("xgb", "rdkit"), ("xgb", "ecfp")],
    "Consensus 3": [("xgb", "rdkit"), ("xgb", "ecfp"), ("xgb", "maccs")],
    "Consensus 4": [("xgb", "rdkit"), ("xgb", "ecfp"), ("xgb", "maccs"), ("bilstm", "rdkit")],
    "Consensus 5": [("xgb", "rdkit"), ("xgb", "ecfp"), ("xgb", "maccs"), ("bilstm", "rdkit"), ("bilstm", "ecfp")],
    "Consensus 6": [("xgb", "rdkit"), ("xgb", "ecfp"), ("xgb", "maccs"), ("bilstm", "rdkit"), ("bilstm", "ecfp"), ("bilstm", "maccs")]
}

# Danh sách lưu kết quả để xuất ra CSV
results_summary = []

# === 2. Tính toán các metrics và xuất file .svg & .csv ===
for config_name, model_list in configs_to_plot.items():
    run_briers = []
    run_intercepts = []
    run_slopes = []
    run_icis = []
    run_ensemble_probs = []
    y_true_ref = None
    
    success = True
    # Duyệt qua từng run (1, 2, 3) để tính toán độc lập
    for run_idx in [1, 2, 3]:
        prob_sum = None
        count = 0
        
        for algo, ft in model_list:
            folder = xgb_dir if algo.lower() == "xgb" else bilstm_dir
            path = os.path.join(folder, f"InFlam_full_test_prob_{ft}_run{run_idx}.csv")
            
            if not os.path.exists(path):
                success = False
                break
                
            df_m = pd.read_csv(path).dropna(subset=["y_true", "y_prob"])
            
            if prob_sum is None:
                prob_sum = df_m['y_prob'].values
                y_true_ref = df_m['y_true'].values
            else:
                prob_sum += df_m['y_prob'].values
            count += 1
            
        if not success or count == 0:
            break
            
        # Ensemble probability cho run này
        mean_run_prob = prob_sum / count
        run_ensemble_probs.append(mean_run_prob)
        
        # 1. Brier score
        brier_run = brier_score_loss(y_true_ref, mean_run_prob)
        run_briers.append(brier_run)
        
        # Chuẩn bị dữ liệu cho Intercept & Slope (tránh log(0))
        eps = 1e-15
        p_clipped = np.clip(mean_run_prob, eps, 1 - eps)
        logit_p = np.log(p_clipped / (1 - p_clipped)).reshape(-1, 1)
        
        # 2. Calibration Intercept & Slope
        lr_full = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
        lr_full.fit(logit_p, y_true_ref)
        
        intercept_run = lr_full.intercept_[0]
        slope_run = lr_full.coef_[0][0]
        
        run_intercepts.append(intercept_run)
        run_slopes.append(slope_run)
        
        # 3. Integrated Calibration Index (ICI)
        sorted_idx = np.argsort(mean_run_prob)
        p_sorted = mean_run_prob[sorted_idx]
        y_sorted = y_true_ref[sorted_idx]
        
        smoothed = lowess(y_sorted, p_sorted, return_sorted=False, frac=0.3)
        ici_run = np.mean(np.abs(p_sorted - smoothed))
        run_icis.append(ici_run)
        
    if success and len(run_briers) == 3:
        # Tính Mean và SD qua 3 runs
        brier_mean, brier_std = np.mean(run_briers), np.std(run_briers)
        intercept_mean, intercept_std = np.mean(run_intercepts), np.std(run_intercepts)
        slope_mean, slope_std = np.mean(run_slopes), np.std(run_slopes)
        ici_mean, ici_std = np.mean(run_icis), np.std(run_icis)
        
        brier_text_sd = f"{brier_mean:.3f} ± {brier_std:.3f}"
        slope_text_sd = f"{slope_mean:.3f} ± {slope_std:.3f}"
        
        # Lưu vào danh sách tổng hợp
        results_summary.append({
            "Configuration": config_name,
            "Brier_Score": f"{brier_mean:.3f} ± {brier_std:.3f}",
            "Calibration_Intercept": f"{intercept_mean:.3f} ± {intercept_std:.3f}",
            "Calibration_Slope": f"{slope_mean:.3f} ± {slope_std:.3f}",
            "ICI": f"{ici_mean:.3f} ± {ici_std:.3f}",
            "Brier_Mean": brier_mean, "Brier_Std": brier_std,
            "Intercept_Mean": intercept_mean, "Intercept_Std": intercept_std,
            "Slope_Mean": slope_mean, "Slope_Std": slope_std,
            "ICI_Mean": ici_mean, "ICI_Std": ici_std
        })
        
        # Lấy xác suất trung bình của 3 runs để vẽ biểu đồ
        mean_ensemble_prob = np.mean(run_ensemble_probs, axis=0)
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true_ref, mean_ensemble_prob, n_bins=10, strategy='uniform'
        )
        
        # Vẽ biểu đồ
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", alpha=0.7)
        
        # Thêm Calibration Slope vào nhãn (legend) bên cạnh Brier score
        label_text = f"{config_name}\nBrier score = {brier_text_sd}\nCalibration slope = {slope_text_sd}"
        plt.plot(mean_predicted_value, fraction_of_positives, marker='o', linewidth=2, color='b', label=label_text)
        
        plt.xlabel("Mean predicted probability", fontsize=14, fontweight='bold', fontstyle="italic")
        plt.ylabel("Fraction of positives", fontsize=14, fontweight='bold', fontstyle="italic")
        
        plt.legend(loc="lower right", fontsize=13, frameon=True)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        
        plt.tight_layout()
        
        safe_name = config_name.replace(' ', '_').replace('+', 'and')
        filename = f"calibration_{safe_name}.svg"
        
        # Lưu vào thư mục chứa code
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        svg_filepath = os.path.join(current_script_dir, filename)
        
        plt.savefig(svg_filepath, format='svg', dpi=300)
        plt.close()
        
        print(f"Đã xuất biểu đồ: {svg_filepath}")
        print(f"  -> Brier Score: {brier_mean:.3f} ± {brier_std:.3f}")
        print(f"  -> Calibration Slope: {slope_mean:.3f} ± {slope_std:.3f}\n")
    else:
        print(f"[{config_name}] Thiếu dữ liệu file ở một số run, bỏ qua.")

# === 3. Xuất toàn bộ kết quả ra file CSV tại thư mục chứa code ===
if results_summary:
    df_results = pd.DataFrame(results_summary)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_filename = os.path.join(current_script_dir, "consensus_calibration_metrics.csv")
    
    df_results.to_csv(csv_filename, index=False)
    print(f"\nĐã lưu thành công file CSV tại: {csv_filename}")

print("\nHoàn tất!")