import pandas as pd

# 1. Đọc file raw vừa xuất ra
df_raw = pd.read_csv("InFlam_full_XGB_AD_k2_10_metrics_raw.csv")

# 2. Định nghĩa các metrics cần tính Mean ± SD
metrics_to_summary = [
    "AUROC",
    "AUPRC",
    "MCC",
    "Accuracy",
    "Balanced Accuracy",
    "Precision",
    "Sensitivity",
    "Specificity",
    "F1",
    "Coverage(%)",
    "N_Samples_Kept",
]

# 3. Gom nhóm theo Fingerprint và k_neighbor để tính Mean và Std
grouped = df_raw.groupby(["Fingerprint", "k_neighbor"])

summary_mean = grouped[metrics_to_summary].mean()
summary_std = grouped[metrics_to_summary].std().fillna(0)

# 4. Gộp dạng "Mean ± SD"
df_summary = pd.DataFrame(index=summary_mean.index)

for col in metrics_to_summary:
  mean_val = summary_mean[col]
  std_val = summary_std[col]
  # Làm tròn 3 chữ số thập phân cho đẹp
  df_summary[col] = (
      mean_val.map(lambda x: f"{x:.3f}")
      + " ± "
      + std_val.map(lambda x: f"{x:.3f}")
  )

df_summary = df_summary.reset_index()

# 5. Xuất file tổng hợp Mean ± SD
output_summary_csv = "InFlam_full_XGB_AD_k2_10_metrics_summary_mean_sd.csv"
df_summary.to_csv(output_summary_csv, index=False, encoding="utf-8-sig")

print(f"✅ Đã xuất bảng tổng hợp Mean ± SD tại: {output_summary_csv}")
print(df_summary.head(10).to_string(index=False))