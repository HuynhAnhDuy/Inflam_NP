import pandas as pd

# 1. Read the 3 CSV files
df1 = pd.read_csv("1.Inflampred_preprocess.csv")
df2 = pd.read_csv("2.AISMPred_preprocess.csv")
df3 = pd.read_csv("InflamNat_NO_master_clean_unique.csv")

# Verify required columns
required_cols = ["y_label", "canonical_smiles"]
file_mapping = {
    "1.Inflampred_preprocess.csv": df1,
    "2.AISMPred_preprocess.csv": df2,
    "InflamNat_NO_master_clean_unique.csv": df3,
}

for name, df in file_mapping.items():
  for col in required_cols:
    if col not in df.columns:
      raise ValueError(f"File {name} is missing required column: '{col}'")

# Lưu tổng số record gốc của file 3 trước khi lọc để tiện thống kê
total_df3_original = len(df3)

# Đếm số mẫu âm (y_label = 0 hoặc 0.0 hoặc "0") trong file 3 ban đầu bị loại bỏ
negative_df3 = df3[df3["y_label"].isin([0, 0.0, "0"])]
count_negative_removed = len(negative_df3)

# 2. CHỈ LẤY y_label = 1 trong file 3
df3_filtered = df3[df3["y_label"].isin([1, 1.0, "1"])]
count_positive_kept_initial = len(df3_filtered)

# 3. Combine all data into a single DataFrame with source tracking
df1["source"] = "file1"
df2["source"] = "file2"
df3_filtered["source"] = "file3"

combined_df = pd.concat([df1, df2, df3_filtered], ignore_index=True)

# 4. Analyze duplicates and conflicting labels based on canonical_smiles
smiles_group = combined_df.groupby("canonical_smiles")["y_label"].unique()

# Identify smiles with conflicting labels (more than 1 unique y_label for the same smiles)
conflicting_smiles = set(
    smiles_group[smiles_group.apply(lambda x: len(x) > 1)].index
)

# 5. Filter out conflicting samples from the merged dataset
cleaned_df = combined_df[~combined_df["canonical_smiles"].isin(conflicting_smiles)]

# Đếm xem trong số các mẫu y_label = 1 của file 3, có bao nhiêu mẫu bị loại do conflict với file 1 hoặc file 2
df3_in_cleaned = cleaned_df[cleaned_df["source"] == "file3"]
count_positive_final_from_file3 = len(df3_in_cleaned.drop_duplicates(subset=["canonical_smiles"]))
count_file3_conflicting_removed = count_positive_kept_initial - count_positive_final_from_file3

# Giữ lại các cột cần thiết cho output cuối cùng
columns_to_keep = ["y_label", "canonical_smiles"]
if "Name" in cleaned_df.columns:
  columns_to_keep.insert(0, "Name")

final_output_df = cleaned_df[columns_to_keep].drop_duplicates()

# 6. Export to a new CSV file
output_filename = "merged_clean_dataset.csv"
final_output_df.to_csv(output_filename, index=False)

# 7. Print detailed statistics to terminal
total_duplicates_or_conflicts = len(combined_df) - len(final_output_df)
conflicting_count = len(conflicting_smiles)

print("=== MERGING & CLEANING SUMMARY ===")
print(f"- Total records in File 1: {len(df1)}")
print(f"- Total records in File 2: {len(df2)}")
print(f"- Total records in File 3 (Original): {total_df3_original}")
print(f"  + Negative samples (y_label=0) removed from File 3: {count_negative_removed}")
print(f"  + Positive samples (y_label=1) initially selected from File 3: {count_positive_kept_initial}")
print(f"  + Positive samples from File 3 removed due to label conflicts with File 1/2: {count_file3_conflicting_removed}")
print(f"  + Positive samples finally kept from File 3: {count_positive_final_from_file3}")
print("-" * 40)
print(f"- Total records combined (before cleaning): {len(combined_df)}")
print(f"- Number of unique SMILES with conflicting labels removed: {conflicting_count}")
print(f"- Total records removed (duplicates & conflicts): {total_duplicates_or_conflicts}")
print(f"- Final records in merged output: {len(final_output_df)}")
print(f"-> Saved final dataset to '{output_filename}'")