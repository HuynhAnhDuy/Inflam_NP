import pandas as pd
from functools import reduce

# ====== CONFIG ======
FILES = [
    "/home/andy/andy/Inflam_NP/NP_predictions/NPASS_test_pred_BiLSTM_ecfp.csv",
    "/home/andy/andy/Inflam_NP/NP_predictions/NPASS_test_pred_BiLSTM_maccs.csv",
    "/home/andy/andy/Inflam_NP/NP_predictions/NPASS_test_pred_BiLSTM_rdkit.csv",
    "/home/andy/andy/Inflam_NP/NP_predictions/NPASS_test_pred_XGB_ecfp.csv",
    "/home/andy/andy/Inflam_NP/NP_predictions/NPASS_test_pred_XGB_maccs.csv",
    "/home/andy/andy/Inflam_NP/NP_predictions/NPASS_test_pred_XGB_rdkit.csv",
]
OUTPUT = "NPASS_candidates.csv"
KEY = "canonical_smiles"   # cột khóa chung
# ====================

dfs = []

for f in FILES:
    df = pd.read_csv(f)
    # Lọc y_pred = 1
    df = df[df["y_pred"] == 1]
    # Chỉ lấy cột canonical_smiles
    dfs.append(df[[KEY]])

# Lấy giao (intersection) giữa tất cả các file
df_common = reduce(lambda left, right: pd.merge(left, right, on=KEY, how="inner"), dfs)

# Ghép với file gốc số 1 để lấy thêm cột A, B, y_pred
df1 = pd.read_csv(FILES[0])
df1_y1 = df1[df1["y_pred"] == 1]

# Merge để lấy canonical_smiles + A + B + y_pred
result = pd.merge(df_common, df1_y1[["ID",KEY, "lipinski_rule_of_five_violations", "sa_score", "y_pred"]], on=KEY, how="inner")

# Xuất kết quả
result.to_csv(OUTPUT, index=False)

print(f"Tìm thấy {len(result)} mẫu chung có y_pred=1 trong cả {len(FILES)} file")
print(f"Đã xuất ra file {OUTPUT}")
