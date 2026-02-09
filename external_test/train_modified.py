import pandas as pd

# ====== FILE PATHS ======
FILE1 = "/home/andy/andy/Inflam_NP/paper/InFlam_full.csv"   # thay bằng đường dẫn của bạn
FILE2 = "/home/andy/andy/Inflam_NP/paper/NPASS_candidates_final_304.csv"
# ========================

# Đọc 2 file
df1 = pd.read_csv(FILE1)
df2 = pd.read_csv(FILE2)

# Lấy cột canonical_smiles
s1 = df1["canonical_smiles"].dropna().astype(str)
s2 = df2["canonical_smiles"].dropna().astype(str)

# Tìm giao giữa 2 tập SMILES
common_smiles = set(s1) & set(s2)

print("=== Duplicate Check Between Two Files ===")
print(f"File 1 total SMILES : {len(s1)}")
print(f"File 2 total SMILES : {len(s2)}")
print(f"Common SMILES count : {len(common_smiles)}")

# Nếu muốn xem các SMILES trùng:
print("\n=== List of Common SMILES ===")
for smi in sorted(common_smiles):
    print(smi)
