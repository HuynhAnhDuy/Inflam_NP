import pandas as pd

# ==== 🔧 CẤU HÌNH DỮ LIỆU Ở ĐÂY ====
file1 = '/home/andy/andy/Inflam_NP/visualization/Prob_InFlam_full/Prob_2025-09-03_10-13-36_BiLSTM/InFlam_full_test_prob_rdkit_run1.csv'    # Thay bằng tên file đầu vào 1
file2 = '/home/andy/andy/Inflam_NP/visualization/Prob_InFlam_full/Prob_2025-09-03_10-13-36_BiLSTM/InFlam_full_test_prob_rdkit_run2.csv'    # Thay bằng tên file đầu vào 2
file3 = '/home/andy/andy/Inflam_NP/visualization/Prob_InFlam_full/Prob_2025-09-03_10-13-36_BiLSTM/InFlam_full_test_prob_rdkit_run3.csv'    # Thay bằng tên file đầu vào 3
output_file = 'BiLSTM_RDKIT_prob.csv'  # Tên file đầu ra
model_name = 'BiLSTM_RDKIT'       # Tên mô hình sẽ ghi vào cột 'model'
# ===================================

# Đọc dữ liệu từ 3 file CSV
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)

# Kiểm tra xem cột y_true có giống nhau không
if not (df1['y_true'].equals(df2['y_true']) and df1['y_true'].equals(df3['y_true'])):
    raise ValueError("⚠️ Các file không có cùng giá trị y_true!")

# Tính trung bình y_prob
avg_y_prob = (df1['y_prob'] + df2['y_prob'] + df3['y_prob']) / 3

# Tạo DataFrame đầu ra
output_df = pd.DataFrame({
    'y_true': df1['y_true'],
    'y_prob': avg_y_prob,
    'model': model_name
})

# Lưu file CSV
output_df.to_csv(output_file, index=False)
print(f"✅ Đã tạo file: {output_file}")
