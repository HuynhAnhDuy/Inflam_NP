import pandas as pd

# ===============================
# Bước 1: Đọc dữ liệu gốc
# ===============================
df1 = pd.read_csv("1.Inflampred_preprocess.csv")
df2 = pd.read_csv("2.AISMPred_preprocess.csv")
df3 = pd.read_csv("3.InFlamNat_preprocess.csv")

key_col = "canonical_smiles"

# Kiểm tra cột tồn tại
for name, df in [("Inflampred", df1), ("AISMPred", df2), ("InFlamNat", df3)]:
    if key_col not in df.columns:
        raise ValueError(f"[{name}] không có cột '{key_col}'")

# ===============================
# Bước 2: Loại khỏi data1 mọi mẫu trùng với data2 hoặc data3
# ===============================
set2 = set(df2[key_col].dropna().unique())
set3 = set(df3[key_col].dropna().unique())
remove_from_1 = set2.union(set3)

# Các mẫu của data1 trùng với data2 / data3 (để lưu log)
overlap1_with2 = df1[df1[key_col].isin(set2)]
overlap1_with3 = df1[df1[key_col].isin(set3)]

# Bản modified cho data1
df1_mod = df1[~df1[key_col].isin(remove_from_1)]

print("Số mẫu Inflampred trùng AISMPred:", len(overlap1_with2))
print("Số mẫu Inflampred trùng InFlamNat:", len(overlap1_with3))
print("Số mẫu Inflampred còn lại sau khi loại trùng (với AISMPred & InFlamNat):", len(df1_mod))

# Lưu kết quả & log
df1_mod.to_csv("1.Inflampred_modified.csv", index=False)
overlap1_with2.to_csv("1.Inflampred_overlap_AISMPred.csv", index=False)
overlap1_with3.to_csv("1.Inflampred_overlap_InFlamNat.csv", index=False)

# ===============================
# Bước 3: Loại khỏi data2 mọi mẫu trùng với data3
# ===============================
set3_only = set(df3[key_col].dropna().unique())
overlap2_with3 = df2[df2[key_col].isin(set3_only)]
df2_mod = df2[~df2[key_col].isin(set3_only)]

print("Số mẫu AISMPred trùng InFlamNat:", len(overlap2_with3))
print("Số mẫu AISMPred còn lại sau khi loại trùng với InFlamNat:", len(df2_mod))

# Lưu kết quả & log
df2_mod.to_csv("2.AISMPred_modified.csv", index=False)
overlap2_with3.to_csv("2.AISMPred_overlap_InFlamNat.csv", index=False)

# (Không chỉnh sửa data3 ở bước này, theo yêu cầu)

# ===============================
# Tóm tắt
# ===============================
print("\n--- TÓM TẮT ---")
print("Inflampred ban đầu:", len(df1))
print("Inflampred sau khi loại trùng với AISMPred & InFlamNat:", len(df1_mod))
print("AISMPred ban đầu:", len(df2))
print("AISMPred sau khi loại trùng với InFlamNat:", len(df2_mod))
print("InFlamNat ban đầu:", len(df3))
