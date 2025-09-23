import pandas as pd

# Load summary files
df_full = pd.read_csv("/home/andy/andy/Inflam_NP/Scaffold_identify/shap_scaffold_split_XGB_20250923_131651/XGB_shap_compounds_carcinogen.csv")
df_test = pd.read_csv("/home/andy/andy/Inflam_NP/Scaffold_identify/shap_scaffold_split_XGB_20250923_131651/XGB_shap_compounds_skin_toxicity.csv")

# Chỉ lấy hàng Assessment = "safety"
df_full_pos = df_full[df_full["Assessment"] == "safety"]
df_test_pos = df_test[df_test["Assessment"] == "safety"]

# Giao giữa 2 tập theo cột Input-SMILES
common_pos = pd.merge(df_full_pos, df_test_pos, on="Input_SMILES")

# Lưu kết quả
common_pos.to_csv("XGB_shap_compounds_safety.csv", index=False)

print(f"✅ Tìm thấy {len(common_pos)} compounds an toàn chung giữa 2 datasets.")
print("💾 File lưu: XGB_shap_compounds_safety.csv")
