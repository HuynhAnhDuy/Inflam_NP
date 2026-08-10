import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Đường dẫn thư mục gốc chứa dữ liệu của bạn
dir_path = r"D:\Andy\Inflam_NP\Predictive_models\Prob_InFlam_in_ad\Prob_2026-08-10_BiLSTM"

# Tạo thư mục riêng để lưu kết quả Uncertainty
output_dir = os.path.join(dir_path, "uncertainty_estimation")
os.makedirs(output_dir, exist_ok=True)

feature_types = ["ecfp", "maccs", "rdkit"]
uncertainty_summary = []

print(
    f"=== TÍNH TOÁN UNCERTAINTY ESTIMATION (XGBoost) [Lưu tại: {output_dir}] ==-\n"
)

for ft in feature_types:
    print(f"--- Đang xử lý Feature: [{ft.upper()}] ---")

    # 1. Đọc xác suất từ 3 runs
    prob1_path = os.path.join(dir_path, f"InFlam_in_ad_test_prob_{ft}_run1.csv")
    prob2_path = os.path.join(dir_path, f"InFlam_in_ad_test_prob_{ft}_run2.csv")
    prob3_path = os.path.join(dir_path, f"InFlam_in_ad_test_prob_{ft}_run3.csv")

    prob_run1 = pd.read_csv(prob1_path).values[:, -1]
    prob_run2 = pd.read_csv(prob2_path).values[:, -1]
    prob_run3 = pd.read_csv(prob3_path).values[:, -1]

    # Gộp xác suất 3 runs thành ma trận (số mẫu x 3)
    prob_matrix = np.column_stack([prob_run1, prob_run2, prob_run3])

    # 2. Tính độ bất định bằng độ lệch chuẩn (Standard Deviation) qua 3 runs
    uncertainty_std = np.std(prob_matrix, axis=1)
    mean_prob = np.mean(prob_matrix, axis=1)

    # Lưu kết quả chi tiết cho từng phân tử ra file CSV riêng
    df_sample_uncertainty = pd.DataFrame(
        {
            "Mean_Probability": mean_prob,
            "Uncertainty_Std": uncertainty_std,
        }
    )
    sample_csv_path = os.path.join(output_dir, f"uncertainty_samples_{ft}.csv")
    df_sample_uncertainty.to_csv(sample_csv_path, index=False)

    # 3. Tính các chỉ số thống kê tổng hợp
    mean_unc = np.mean(uncertainty_std)
    median_unc = np.median(uncertainty_std)
    max_unc = np.max(uncertainty_std)

    print(f"  > Mean Uncertainty (Std): {mean_unc:.4f} | Max: {max_unc:.4f}")

    uncertainty_summary.append(
        {
            "Feature": ft.upper(),
            "Mean_Uncertainty_Std": round(mean_unc, 4),
            "Median_Uncertainty_Std": round(median_unc, 4),
            "Max_Uncertainty_Std": round(max_unc, 4),
        }
    )

    # 4. Vẽ biểu đồ phân phối độ bất định
    plt.figure(figsize=(7, 5))
    sns.histplot(uncertainty_std, kde=True, color="#2b5c8f", bins=20)
    plt.title(
        f"Prediction Uncertainty Distribution - XGBoost ({ft.upper()})",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel(
        "Standard deviation across 3 runs (Uncertainty)",
        fontsize=11,
        fontweight="bold",
    )
    plt.ylabel("Number of molecules", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    fig_path = os.path.join(output_dir, f"uncertainty_dist_xgb_{ft}.svg")
    plt.savefig(fig_path, format="svg", dpi=300)
    plt.close()

    print(f"  > Đã lưu biểu đồ tại: {fig_path}")
    print("-" * 50)

# Xuất bảng tổng hợp độ bất định ra file CSV chung
df_unc_summary = pd.DataFrame(uncertainty_summary)
summary_csv_path = os.path.join(output_dir, "uncertainty_summary.csv")
df_unc_summary.to_csv(summary_csv_path, index=False)

print(f"\nHoàn tất! Đã lưu bảng tổng hợp Uncertainty vào: {summary_csv_path}")