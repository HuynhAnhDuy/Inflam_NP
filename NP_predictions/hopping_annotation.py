import pandas as pd

# ===== CONFIG =====
FILE1   = "/home/andy/andy/Inflam_NP/NP_predictions/NPASS_common_scaffold_hopping_3features.csv"   # chứa canonical_smiles
FILE2   = "/home/andy/andy/Inflam_NP/NP_predictions/NPASS_candidates_final_304.csv"          # chứa canonical_smiles, ID, lipinski_rule_of_five_violations, sa_score
OUTPUT  = "NPASS_common_scaffold_hopping_annotated_2.csv"

# ===== LOAD DATA =====
df1 = pd.read_csv(FILE1)
df2 = pd.read_csv(FILE2)

# ===== MERGE =====
# merge dựa trên canonical_smiles
df_merged = pd.merge(
    df1,
    df2[["canonical_smiles", "ID", "lipinski_rule_of_five_violations", "sa_score","prob_avg"]],
    on="canonical_smiles",
    how="left"   # giữ tất cả rows của file1
)

# ===== SAVE =====
df_merged.to_csv(OUTPUT, index=False)

print(f"Xuất file {OUTPUT} với {len(df_merged)} dòng.")
