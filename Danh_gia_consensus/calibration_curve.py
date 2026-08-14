import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# === 1. Cấu hình đường dẫn ===
xgb_dir = r"D:\Andy\Inflam_NP\Predictive_models\Prob_InFlam_Full\Prob_2026-08-10_xgb"
bilstm_dir = r"D:\Andy\Inflam_NP\Predictive_models\Prob_InFlam_Full\Prob_2026-08-10_BiLSTM"

# Định nghĩa 6 cấu hình bạn yêu cầu
configs_to_plot = {
    "XGB_RDKIT": [("xgb", "rdkit")],
    "Combo 2": [("xgb", "rdkit"), ("xgb", "ecfp")],
    "Combo 3": [("xgb", "rdkit"), ("xgb", "ecfp"), ("xgb", "maccs")],
    "Combo 4": [("xgb", "rdkit"), ("xgb", "ecfp"), ("xgb", "maccs"), ("bilstm", "rdkit")],
    "Combo 5": [("xgb", "rdkit"), ("xgb", "ecfp"), ("xgb", "maccs"), ("bilstm", "rdkit"), ("bilstm", "ecfp")],
    "Combo 6": [("xgb", "rdkit"), ("xgb", "ecfp"), ("xgb", "maccs"), ("bilstm", "rdkit"), ("bilstm", "ecfp"), ("bilstm", "maccs")]
}

# === 2. Tính toán, lấy Brier Score dạng Mean ± SD và xuất từng file .svg riêng biệt ===
for config_name, model_list in configs_to_plot.items():
    run_briers = []
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
        
        # Tính Brier score cho riêng run này
        brier_run = brier_score_loss(y_true_ref, mean_run_prob)
        run_briers.append(brier_run)
        
    if success and len(run_briers) == 3:
        # Tính Mean và SD của Brier Score qua 3 runs
        brier_mean = np.mean(run_briers)
        brier_std = np.std(run_briers)
        brier_text_sd = f"{brier_mean:.3f} ± {brier_std:.3f}"
        
        # Lấy xác suất trung bình của 3 runs (để vẽ đường cong mượt mà đại diện)
        mean_ensemble_prob = np.mean(run_ensemble_probs, axis=0)
        
        # Tính toán điểm trên calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true_ref, mean_ensemble_prob, n_bins=10, strategy='uniform'
        )
        
        # Vẽ biểu đồ cho từng cấu hình
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", alpha=0.7)
        
        # Đưa Brier Score dạng Mean ± SD vào nhãn (legend) của đường vẽ
        label_text = f"{config_name} (Brier score = {brier_text_sd})"
        plt.plot(mean_predicted_value, fraction_of_positives, marker='o', linewidth=2, color='b', label=label_text)
        
        plt.xlabel("Mean predicted probability", fontsize=14, fontweight='bold',fontstyle="italic")
        plt.ylabel("Fraction of positives", fontsize=14, fontweight='bold',fontstyle="italic")
        
        plt.legend(loc="lower right", fontsize=12, frameon=True)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        
        plt.tight_layout()
        
        # Đặt tên file an toàn và lưu dạng SVG
        safe_name = config_name.replace(' ', '_').replace('+', 'and')
        filename = f"calibration_{safe_name}.svg"
        plt.savefig(filename, format='svg', dpi=300)
        plt.close() # Đóng figure để giải phóng bộ nhớ
        
        print(f"Đã xuất thành công: {filename} (Brier Score: {brier_text_sd})")
    else:
        print(f"[{config_name}] Thiếu dữ liệu file ở một số run, bỏ qua vẽ biểu đồ này.")

print("\nHoàn tất! Đã tạo và lưu đủ các file hình ảnh SVG có kèm Brier Score dạng Mean ± SD.")