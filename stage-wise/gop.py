import pandas as pd

# 1. Đọc hai file dữ liệu
df_main = pd.read_csv("full_processed_data_SA2_affinity.csv")
df_x = pd.read_csv("x.csv")

# Đảm bảo cột Index có cùng kiểu dữ liệu (tránh lỗi lệch kiểu int/str) để match chính xác
df_main['Index'] = df_main['Index'].astype(str)
df_x['Index'] = df_x['Index'].astype(str)

# 2. Merge (kết hợp) dữ liệu dựa vào cột 'Index' (sử dụng left join để giữ nguyên toàn bộ file gốc)
df_merged = pd.merge(df_main, df_x[['Index', 'Druglikeness']], on='Index', how='left')

# 3. Điền giá trị 'no' cho các mẫu không match được (các ô bị NaN)
df_merged['Druglikeness'] = df_merged['Druglikeness'].fillna('no')

# 4. Lưu lại file kết quả (ghi đè file gốc hoặc lưu ra file mới tùy bạn chọn)
output_filename = "full_processed_data_SA2_affinity_2.csv"
df_merged.to_csv(output_filename, index=False)

print(f"Đã tích hợp thành công! File mới đã được lưu vào: {output_filename}")
print(f"Số lượng mẫu sau khi tích hợp: {len(df_merged)}")
print(f"Số lượng mẫu có Druglikeness khác 'no': (df_merged['Druglikeness'] != 'no').sum()")