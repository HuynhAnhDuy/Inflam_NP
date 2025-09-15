import pandas as pd

# Load summary files
df_full = pd.read_csv("scaffold_shap_summary_full.csv")
df_test = pd.read_csv("scaffold_shap_summary_test.csv")

# Chỉ lấy scaffold positive
df_full_pos = df_full[df_full["effect"] == "positive"]
df_test_pos = df_test[df_test["effect"] == "positive"]

# Giao giữa 2 tập scaffold positive
common_pos = pd.merge(df_full_pos, df_test_pos, on="scaffold", suffixes=("_full", "_test"))

# Lưu kết quả
common_pos.to_csv("scaffold_positive_overlap.csv", index=False)

print(f"✅ Tìm thấy {len(common_pos)} scaffolds positive chung giữa full & test.")
print("💾 Lưu file scaffold_positive_overlap.csv")
