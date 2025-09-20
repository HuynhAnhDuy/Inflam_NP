import pandas as pd
from functools import reduce

# ====== CONFIG ======
FILES = [
    "/home/andy/andy/Inflam_NP/Predictive_models/Prob_NPASS/NPASS_test_pred_BiLSTM_ecfp.csv",
    "/home/andy/andy/Inflam_NP/Predictive_models/Prob_NPASS/NPASS_test_pred_BiLSTM_maccs.csv",
    "/home/andy/andy/Inflam_NP/Predictive_models/Prob_NPASS/NPASS_test_pred_BiLSTM_rdkit.csv",
    "/home/andy/andy/Inflam_NP/Predictive_models/Prob_NPASS/NPASS_test_pred_XGB_ecfp.csv",
    "/home/andy/andy/Inflam_NP/Predictive_models/Prob_NPASS/NPASS_test_pred_XGB_maccs.csv",
    "/home/andy/andy/Inflam_NP/Predictive_models/Prob_NPASS/NPASS_test_pred_XGB_rdkit.csv",
]
OUTPUT = "NPASS_candidate_y1.csv"
KEY = "canonical_smiles"   # cột khóa chung
# ====================

dfs = []
probs = []  # lưu y_pro_average để tính trung bình

for i, f in enumerate(FILES):
    df = pd.read_csv(f)
    # Lọc y_pred = 1
    df = df[df["y_pred"] == 1].copy()
    # Đổi tên cột y_pro_average để phân biệt từng file
    df.rename(columns={"y_pro_average": f"prob_{i+1}"}, inplace=True)
    dfs.append(df[[KEY]])
    probs.append(df[[KEY, f"prob_{i+1}"]])

# Lấy giao (intersection) giữa tất cả các file
df_common = reduce(lambda left, right: pd.merge(left, right, on=KEY, how="inner"), dfs)

# Ghép toàn bộ prob từ 6 file vào
for p in probs:
    df_common = pd.merge(df_common, p, on=KEY, how="inner")

# Tính trung bình của 6 prob
prob_cols = [f"prob_{i+1}" for i in range(len(FILES))]
df_common["prob_avg"] = df_common[prob_cols].mean(axis=1)

# Ghép với file gốc số 1 để lấy thêm cột ID, lipinski, sa_score, y_pred
df1 = pd.read_csv(FILES[0])
df1_y1 = df1[df1["y_pred"] == 1]
result = pd.merge(
    df_common,
    df1_y1[["ID", KEY, "lipinski_rule_of_five_violations", "sa_score", "y_pred"]],
    on=KEY,
    how="inner"
)

# Xuất kết quả
result.to_csv(OUTPUT, index=False)

print(f"Tìm thấy {len(result)} mẫu chung có y_pred=1 trong cả {len(FILES)} file")
print(f"Đã xuất ra file {OUTPUT}")
