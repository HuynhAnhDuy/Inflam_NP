import pandas as pd

df = pd.read_csv("InFlam_full.csv")
df.columns = df.columns.str.strip()

df = df[["canonical_smiles", "Label"]].dropna()
df["canonical_smiles"] = df["canonical_smiles"].astype(str).str.strip()
df["Label"] = df["Label"].astype(str).str.strip()

# Tìm SMILES có nhiều hơn 1 label
conflicting_smiles = (
    df.groupby("canonical_smiles")["Label"]
    .nunique()
    .loc[lambda x: x > 1]
    .index
)

# Đếm số conflicting rows
conflicting_records = df[df["canonical_smiles"].isin(conflicting_smiles)]

print("Number of conflicting SMILES:", len(conflicting_smiles))
print("Number of conflicting records:", len(conflicting_records))