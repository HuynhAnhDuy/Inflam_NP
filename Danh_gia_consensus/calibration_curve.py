import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import brier_score_loss

# Đường dẫn thư mục gốc chứa dữ liệu của bạn
dir_path = r"D:\Andy\Inflam_NP\Predictive_models\Prob_InFlam_in_ad\Prob_2026-08-10_BiLSTM"

# Tạo thư mục riêng để lưu biểu đồ
output_dir = os.path.join(dir_path, "calibration_curve")
os.makedirs(output_dir, exist_ok=True)

feature_types = ["ecfp", "maccs", "rdkit"]
calibration_summary = []

print(
    f"=== TÍNH TOÁN BRIER SCORE VÀ VẼ CALIBRATION CURVES (XGBoost) ==-\n"
)

for ft in feature_types:
    print(f"--- Đang xử lý Feature: [{ft.upper()}] ---")

    # 1. Đọc nhãn thực tế In-AD
    y_test_path = os.path.join(dir_path, f"InFlam_in_ad_y_test.csv")
    if not os.path.exists(y_test_path):
        y_test_path = f"InFlam_in_ad_y_test_{ft}.csv"

    df_y = pd.read_csv(y_test_path)
    y_test_in_ad = df_y["Label"].values.astype(int)

    # 2. Đọc xác suất 3 runs
    prob1_path = os.path.join(dir_path, f"InFlam_in_ad_test_prob_{ft}_run1.csv")
    prob2_path = os.path.join(dir_path, f"InFlam_in_ad_test_prob_{ft}_run2.csv")
    prob3_path = os.path.join(dir_path, f"InFlam_in_ad_test_prob_{ft}_run3.csv")

    prob_run1 = pd.read_csv(prob1_path).values[:, -1]
    prob_run2 = pd.read_csv(prob2_path).values[:, -1]
    prob_run3 = pd.read_csv(prob3_path).values[:, -1]

    # Tính xác suất trung bình qua 3 runs
    mean_probs = (prob_run1 + prob_run2 + prob_run3) / 3.0

    # 3. Tính chỉ số định lượng Brier Score (Càng thấp càng tốt, tối ưu là 0)
    brier_val = brier_score_loss(y_test_in_ad, mean_probs)

    print(f"  > Brier Score (Mean 3 Runs): {brier_val:.3f}")

    # Lưu kết quả vào danh sách tổng hợp
    calibration_summary.append(
        {"Feature": ft.upper(), "Brier_Score": round(brier_val, 3)}
    )

    # 4. Vẽ biểu đồ Calibration Curve
    plt.figure(figsize=(7, 7))
    CalibrationDisplay.from_predictions(
        y_test_in_ad,
        mean_probs,
        n_bins=10,
        name=f"XGBoost ({ft.upper()}) [Brier = {brier_val:.3f}]",
        color="#2b5c8f",
        marker="o",
    )

    plt.title(
        f"Calibration Curve - XGBoost ({ft.upper()})",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel(
        "Mean predicted probability", fontsize=11, fontweight="bold"
    )
    plt.ylabel("Fraction of positives", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    output_fig = os.path.join(output_dir, f"calibration_curve_xgb_{ft}.svg")
    plt.savefig(output_fig, format="svg", dpi=300)
    plt.close()

    print(f"  > Đã lưu biểu đồ tại: {output_fig}")
    print("-" * 50)

# Xuất bảng tổng hợp Brier Score ra file CSV để đưa vào bài báo
df_brier = pd.DataFrame(calibration_summary)
brier_csv_path = os.path.join(output_dir, "brier_score_summary.csv")
df_brier.to_csv(brier_csv_path, index=False)

print(f"\nHoàn tất! Đã lưu bảng tổng hợp Brier Score vào: {brier_csv_path}")