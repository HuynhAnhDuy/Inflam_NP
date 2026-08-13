import pandas as pd

# 1. Đọc 3 file
df2 = pd.read_csv("stage_wise_compounds_for_docking_SA2.csv")
df3 = pd.read_csv("stage_wise_compounds_for_docking_SA3.csv")
df4 = pd.read_csv("stage_wise_compounds_for_docking_SA4.csv")

# 2. Lấy tập các SMILES (sử dụng set để kiểm tra nhanh)
set2 = set(df2['canonical_smiles'])
set3 = set(df3['canonical_smiles'])
set4 = set(df4['canonical_smiles'])

# 3. Kiểm tra độ bao hàm
is_2_in_3 = set2.issubset(set3)
is_3_in_4 = set3.issubset(set4)

print(f"Số lượng hợp chất SA2: {len(set2)}")
print(f"Số lượng hợp chất SA3: {len(set3)}")
print(f"Số lượng hợp chất SA4: {len(set4)}")

print(f"\n--- KẾT QUẢ KIỂM TRA ---")
print(f"Tập SA2 có nằm trong SA3 không? {'CÓ' if is_2_in_3 else 'KHÔNG'}")
print(f"Tập SA3 có nằm trong SA4 không? {'CÓ' if is_3_in_4 else 'KHÔNG'}")

# Nếu có sai lệch, tìm xem chất nào ở SA2 mà không nằm trong SA3
if not is_2_in_3:
    diff = set2 - set3
    print(f"Cảnh báo: Có {len(diff)} chất trong SA2 không nằm trong SA3. Xem mẫu: {list(diff)[:5]}")