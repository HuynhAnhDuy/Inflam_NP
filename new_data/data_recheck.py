import pandas as pd

# 1. Read the 3 CSV files
df1 = pd.read_csv("1.Inflampred_preprocess.csv")
df2 = pd.read_csv("2.AISMPred_preprocess.csv")
df3 = pd.read_csv("InflamNat_NO_master_clean_unique.csv")

# Verify that required columns exist in all files
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

# Kiểm tra xem file 3 có cột "Name" không để lấy dữ liệu
has_name_col = "Name" in df3.columns

# 2. Create lookup dictionaries (mapping) from File 1 and File 2
dict1 = pd.Series(df1.y_label.values, index=df1.canonical_smiles).to_dict()
dict2 = pd.Series(df2.y_label.values, index=df2.canonical_smiles).to_dict()

# 3. Check each record in File 3 against File 1 and File 2
results = []

for idx, row in df3.iterrows():
  smiles = row["canonical_smiles"]
  y_label_3 = row["y_label"]
  name_val = row["Name"] if has_name_col else None

  in_file1 = smiles in dict1
  in_file2 = smiles in dict2
  is_duplicate = in_file1 or in_file2

  match_f1 = dict1.get(smiles, None) if in_file1 else None
  match_f2 = dict2.get(smiles, None) if in_file2 else None

  note = "Not Duplicate"
  label_match = None

  if is_duplicate:
    labels_to_check = []
    if in_file1:
      labels_to_check.append(match_f1)
    if in_file2:
      labels_to_check.append(match_f2)

    # Check if File 3 y_label matches at least one of the existing files
    label_match = y_label_3 in labels_to_check

    if label_match:
      note = "Consistent (Matching y_label)"
    else:
      note = "Conflicting (Different y_label)"

  results.append({
      "Name": name_val,
      "canonical_smiles": smiles,
      "y_label_file3": y_label_3,
      "in_file1": in_file1,
      "y_label_file1": match_f1,
      "in_file2": in_file2,
      "y_label_file2": match_f2,
      "is_duplicate": is_duplicate,
      "y_label_match": label_match,
      "note": note,
  })

df_results = pd.DataFrame(results)

# 4. Separate into duplicate and non-duplicate sets
df_duplicate = df_results[df_results["is_duplicate"] == True]
df_unique = df_results[df_results["is_duplicate"] == False]

# Khôi phục cấu trúc cột cho file không trùng (Name, y_label, canonical_smiles)
if has_name_col:
  df_unique_output = df_unique[["Name", "y_label_file3", "canonical_smiles"]].rename(
      columns={"y_label_file3": "y_label"}
  )
else:
  df_unique_output = df_unique[["y_label_file3", "canonical_smiles"]].rename(
      columns={"y_label_file3": "y_label"}
  )

# 5. Export both results to CSV
output_duplicate = "file3_duplicate_details.csv"
output_unique = "file3_non_duplicate.csv"

df_duplicate.to_csv(output_duplicate, index=False)
df_unique_output.to_csv(output_unique, index=False)

# 6. Print summary statistics to terminal
print("Processing completed!")
print(f"- Total records in File 3: {len(df3)}")
print(
    f"- Total duplicates found: {len(df_duplicate)} (Saved to"
    f" '{output_duplicate}')"
)
print(
    f"- Total non-duplicates (unique): {len(df_unique)} (Saved to"
    f" '{output_unique}')"
)

if len(df_duplicate) > 0:
  consistent_count = (
      df_duplicate["note"] == "Consistent (Matching y_label)"
  ).sum()
  conflicting_count = (
      df_duplicate["note"] == "Conflicting (Different y_label)"
  ).sum()
  print(f"  + Consistent records (matching labels): {consistent_count}")
  print(f"  + Conflicting records (different labels): {conflicting_count}")

# Thống kê số mẫu nhãn 0 trong tập không trùng
if len(df_unique) > 0:
  zero_labels_count = (df_unique["y_label_file3"].isin([0, 0.0, "0"])).sum()
  print(f"Conflicting samples (n= {zero_labels_count})")
else:
  print("Conflicting samples (n= 0)")