import pandas as pd

# 1. Đọc kết quả docking của tập lớn nhất (SA4)
df_res = pd.read_csv("docking_results_SA4.csv")

# 2. Đọc lại 3 file list gốc để lấy 'mask' (bộ lọc)
df2 = pd.read_csv("stage_wise_compounds_for_docking_SA2.csv")
df3 = pd.read_csv("stage_wise_compounds_for_docking_SA3.csv")

# 3. Tạo cột phân loại SA cho kết quả docking
# Mặc định tất cả thuộc nhóm SA4 (đã bao hàm cả 2 và 3)
df_res['SA_Group'] = 'SA4'
# Đánh dấu những chất thuộc SA3 (nhưng > 2)
df_res.loc[df_res['canonical_smiles'].isin(df3['canonical_smiles']), 'SA_Group'] = 'SA3'
# Đánh dấu những chất thuộc SA2
df_res.loc[df_res['canonical_smiles'].isin(df2['canonical_smiles']), 'SA_Group'] = 'SA2'

# 4. Lưu kết quả đã phân nhóm
df_res.to_csv("final_docking_analysis.csv", index=False)

# 5. Thống kê nhanh để xem chất tốt (docking score <= -7.0) nằm ở nhóm nào
good_binders = df_res[df_res['docking_score'] <= -7.0]
print(good_binders.groupby('SA_Group').size())