import pandas as pd

# Read data
x_train = pd.read_csv("/home/andy/andy/Inflam_NP/preprocessing/InFlam_full_x_train.csv")
x_test = pd.read_csv("/home/andy/andy/Inflam_NP/preprocessing/NPASS_filtered_584_safety.csv")

# Find overlapping canonical_smiles
overlap_smiles = set(x_train["canonical_smiles"]) & set(x_test["canonical_smiles"])

# Remove rows in x_train that have canonical_smiles also in x_test
x_train_modified = x_train[~x_train["canonical_smiles"].isin(overlap_smiles)]

# Export to a new file
x_train_modified.to_csv("NPASS_test.csv", index=False)

print(f"Number of samples removed: {len(x_train) - len(x_train_modified)}")
print(f"x_train_modified remaining: {len(x_train_modified)} rows")
