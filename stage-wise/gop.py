import pandas as pd

# 1. Đọc hai file dữ liệu
df_main = pd.read_csv("full_processed_data_SA2.csv")
df_docking = pd.read_csv("COX2_docking_SA2.csv")

# 2. Kiểm tra xem cột 'Index' có cùng kiểu dữ liệu không (tránh lỗi không khớp do một bên là chuỗi, một bên là số)
df_main['Index'] = df_main['Index'].astype(str)
df_docking['Index'] = df_docking['Index'].astype(str)

# 3. Sử dụng merge (tương tự như VLOOKUP trong Excel)
# how='left' giúp giữ lại toàn bộ dòng của file chính, chỉ thêm cột affinity nếu tìm thấy Index khớp
df_merged = pd.merge(df_main, df_docking[['Index', 'affinity (kcal/mol)']], on='Index', how='left')

# 4. Lưu kết quả ra file mới
df_merged.to_csv("full_processed_data_SA2_affinity.csv", index=False)

print("Đã gán thành công giá trị affinity!")
print("File mới đã được lưu là: full_processed_data_with_affinity.csv")