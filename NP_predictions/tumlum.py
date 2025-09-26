import pandas as pd

# Đọc 2 file CSV
file1 = pd.read_csv("/home/andy/andy/Inflam_NP/NP_predictions/NPASS_toxicity_carcinogen_SA2.csv")
file2 = pd.read_csv("/home/andy/andy/Inflam_NP/NP_predictions/NPASS_toxicity_skin_SA2.csv")

# Đảm bảo các cột cần thiết tồn tại
required_cols = {"canonical_smiles", "Assessment"}
if not required_cols.issubset(file1.columns) or not required_cols.issubset(file2.columns):
    raise ValueError("Cả 2 file phải có cột 'canonical_smiles' và 'Assessment'")

# Ghép dữ liệu theo canonical_smiles
merged = pd.merge(
    file1, file2,
    on="canonical_smiles",
    suffixes=("_file1", "_file2")
)

# Tổng số canonical_smiles trùng nhau
total_common = merged.shape[0]

# Số lượng trùng nhau có assessment = "safety" (ở cả hai file)
safety_both = merged[
    (merged["Assessment_file1"].str.lower() == "safety") &
    (merged["Assessment_file2"].str.lower() == "safety")
].shape[0]

# Nếu bạn chỉ muốn check "safety" ở 1 trong 2 file thì có thể đổi điều kiện:
# (merged["assessment_file1"].str.lower() == "safety") |
# (merged["assessment_file2"].str.lower() == "safety")

print("Số lượng canonical_smiles trùng nhau:", total_common)
print("Số lượng trùng nhau có assessment = safety (cả 2 file):", safety_both)
