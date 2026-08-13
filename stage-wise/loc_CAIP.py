import pandas as pd

# 1. Đọc hai file CSV
print("Đang đọc dữ liệu...")
df_test = pd.read_csv("test_day_du.csv")
df_preds = pd.read_csv("CAIP_batch_predictions.csv")

# Danh sách các cột xác suất cần lấy từ file predictions
prob_columns = [
    "canonical_smiles",
    "XGB_ECFP_prob",
    "XGB_MACCS_prob",
    "XGB_RDKIT_prob",
    "BiLSTM_ECFP_prob",
    "BiLSTM_MACCS_prob",
    "BiLSTM_RDKIT_prob"
]

# Kiểm tra xem các cột xác suất có tồn tại trong file predictions không để tránh lỗi
missing_cols = [col for col in prob_columns if col not in df_preds.columns]
if missing_cols:
    raise ValueError(f"Các cột sau không tồn tại trong CAIP_batch_predictions.csv: {missing_cols}")

# Lọc chỉ lấy cột canonical_smiles và các cột prob từ file predictions
df_preds_subset = df_preds[prob_columns]

# 2. Gộp dữ liệu (Merge) dựa trên cột 'canonical_smiles'
# Sử dụng 'left' join để giữ lại toàn bộ dòng của file test_day_du.csv ban đầu
print("Đang tiến hành gộp dữ liệu...")
df_merged = pd.merge(df_test, df_preds_subset, on="canonical_smiles", how="left")

# 3. Xuất ra file CSV output mới
output_filename = "test_day_du_with_predictions.csv"
df_merged.to_csv(output_filename, index=False)

print(f"Thành công! File kết quả đã được lưu tại: {output_filename}")
print(f"Số lượng dòng ban đầu: {len(df_test)} | Số lượng dòng sau khi gộp: {len(df_merged)}")