import pandas as pd

# ====== FILE PATHS ======
FILE_A = "/home/andy/andy/Inflam_NP/paper/NPASS_common_scaffold_hopping_annotated.csv"      # file cần gán label
FILE_B = "/home/andy/andy/Inflam_NP/paper/InFlam_full.csv"      # file gốc chứa canonical_smiles + Label
OUTPUT = "file_A_with_label.csv"
# =========================

# Đọc dữ liệu
dfA = pd.read_csv(FILE_A)
dfB = pd.read_csv(FILE_B)

# Chỉ giữ 2 cột cần thiết ở file B
dfB_subset = dfB[["canonical_smiles", "Label"]].drop_duplicates()

# Merge (left join) theo canonical_smiles
df_out = dfA.merge(dfB_subset,
                   on="canonical_smiles",
                   how="left",
                   suffixes=("", "_original"))

# Đổi tên cột Label thành Original_label
df_out = df_out.rename(columns={"Label": "Original_label"})

# Lưu file
df_out.to_csv(OUTPUT, index=False)

print("Done! File đã lưu:", OUTPUT)
print(df_out.head())
