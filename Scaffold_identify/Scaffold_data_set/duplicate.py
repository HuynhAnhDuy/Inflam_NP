import pandas as pd

# Load scaffold file
df_train_scaf = pd.read_csv("/home/andy/andy/Inflam_NP/Scaffold_identify/Scaffold_data_set/InFlam_full_x_train_with_scaffolds.csv")
df_test_scaf = pd.read_csv("/home/andy/andy/Inflam_NP/Scaffold_identify/Scaffold_data_set/InFlam_full_x_test_with_scaffolds.csv")

# Lấy tập hợp scaffold duy nhất
train_scaffolds = set(df_train_scaf['scaffold'].dropna().unique())
test_scaffolds = set(df_test_scaf['scaffold'].dropna().unique())

# Giao nhau
common_scaffolds = train_scaffolds.intersection(test_scaffolds)

# Tính tỷ lệ
overlap_train = len(common_scaffolds) / len(train_scaffolds) * 100 if len(train_scaffolds) > 0 else 0
overlap_test = len(common_scaffolds) / len(test_scaffolds) * 100 if len(test_scaffolds) > 0 else 0

print(f"🔹 Số scaffold train: {len(train_scaffolds)}")
print(f"🔹 Số scaffold test: {len(test_scaffolds)}")
print(f"🔹 Số scaffold trùng lặp: {len(common_scaffolds)}")
print(f"📊 Tỷ lệ scaffold trùng (so với train): {overlap_train:.2f}%")
print(f"📊 Tỷ lệ scaffold trùng (so với test): {overlap_test:.2f}%")
