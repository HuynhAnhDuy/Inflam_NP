import pandas as pd

# Đường dẫn file CSV
FILE = "/home/andy/andy/Inflam_NP/paper/InFlam_full.csv"   # thay bằng đường dẫn thực tế

# Đọc file
df = pd.read_csv(FILE)

# Kiểm tra số lượng trùng
duplicate_mask = df["canonical_smiles"].duplicated(keep=False)
duplicates = df[df["canonical_smiles"].duplicated(keep=False)]

# In thống kê
total_rows = len(df)
num_unique = df["canonical_smiles"].nunique()
num_duplicates = total_rows - num_unique

print("=== Duplicate Check for canonical_smiles ===")
print(f"Total rows       : {total_rows}")
print(f"Unique SMILES    : {num_unique}")
print(f"Duplicate entries: {num_duplicates}")

# Nếu muốn xem đầy đủ các SMILES trùng:
print("\n=== List of duplicated SMILES ===")
print(duplicates.sort_values("canonical_smiles"))
