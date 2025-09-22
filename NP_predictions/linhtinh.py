import pandas as pd

# Đọc file CSV
df = pd.read_csv("/home/andy/andy/Inflam_NP/NP_predictions/NPASS_common_scaffold_hopping_annotated.csv")

# Tạo cột 'group' (số nguyên, cùng ID -> cùng group)
df["group"] = df.groupby("ID").ngroup()

# Xuất ra file mới
df.to_csv("NPASS_common_scaffold_hopping_annotated_output.csv", index=False)
