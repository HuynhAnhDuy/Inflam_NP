import pandas as pd
import os

# Danh sách các file CSV cần gộp
files = [
    "/home/andy/andy/Inflam_NP/Predictive_models/InFlam_full_metrics/InFlam_full_BiLSTM_fingerprint_metrics_raw.csv",
    "/home/andy/andy/Inflam_NP/Predictive_models/InFlam_full_metrics/InFlam_full_LGBM_fingerprint_metrics_raw.csv",
    "/home/andy/andy/Inflam_NP/Predictive_models/InFlam_full_metrics/InFlam_full_RF_fingerprint_metrics_raw.csv",
    "/home/andy/andy/Inflam_NP/Predictive_models/InFlam_full_metrics/InFlam_full_XGB_fingerprint_metrics_raw.csv"
]

# Đọc và gộp tất cả file (thử latin1 để tránh UnicodeDecodeError)
df_list = [pd.read_csv(f, encoding="latin1") for f in files]
df_full = pd.concat(df_list, ignore_index=True)

# Lấy folder của file đầu tiên trong danh sách
output_folder = os.path.dirname(files[0])

# Đặt tên file output (nằm trong cùng folder với input)
output_file = os.path.join(output_folder, "InFlam_full_all_metrics_raw.csv")

# Xuất ra file CSV mới (xuất lại với utf-8 để chuẩn)
df_full.to_csv(output_file, index=False, encoding="utf-8")

print(f"✅ Đã gộp xong, lưu thành {output_file}")
