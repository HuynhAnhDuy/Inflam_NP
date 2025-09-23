import pandas as pd

# === Config ===
file_scaffolds = "/home/andy/andy/Inflam_NP/Scaffold_identify/shap_scaffold_split_XGB_20250923_131651/scaffold_positive_overlap.csv"  # File 1
file_dataset   = "/home/andy/andy/Inflam_NP/Scaffold_identify/InFlam_full_with_scaffolds.csv"  # File 2
output_counts  = "/home/andy/andy/Inflam_NP/Scaffold_identify/shap_scaffold_split_XGB_20250923_131651/scaffold_positive_overlap_with_counts.csv"
output_details = "/home/andy/andy/Inflam_NP/Scaffold_identify/shap_scaffold_split_XGB_20250923_131651/scaffold_positive_overlap_compound_details.csv"

# === Load data ===
df_scaffolds = pd.read_csv(file_scaffolds)
df_dataset   = pd.read_csv(file_dataset)

# Kiểm tra tên cột
print("File 1 columns:", df_scaffolds.columns.tolist())
print("File 2 columns:", df_dataset.columns.tolist())

# === Chỉ lấy compounds có Label = 1 ===
df_dataset = df_dataset[df_dataset["Label"] == 1].copy()

# === Đếm số compound cho từng scaffold (Label = 1) ===
scaffold_counts = (
    df_dataset.groupby("scaffold")["canonical_smiles"]
    .nunique()  # số compound duy nhất
    .reset_index(name="compound_count")
)

# === Ghép vào file scaffold gốc ===
df_output = df_scaffolds.merge(scaffold_counts, on="scaffold", how="left")
df_output["compound_count"] = df_output["compound_count"].fillna(0).astype(int)

# === Xuất file scaffold + count ===
df_output.to_csv(output_counts, index=False)
print(f"Saved: {output_counts}")

# === Tạo file chi tiết compounds (Label = 1) cho từng scaffold ===
cols_to_keep = ["Index", "canonical_smiles", "Label", "scaffold"]
df_details = df_dataset[cols_to_keep].copy()

# Chỉ lấy compounds thuộc danh sách scaffold trong file 1
df_details = df_details[df_details["scaffold"].isin(df_scaffolds["scaffold"])]

# Xuất ra file chi tiết
df_details.to_csv(output_details, index=False)
print(f"Saved: {output_details}")
