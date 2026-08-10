import os
import re
from functools import reduce
import pandas as pd

# ====== CONFIG ======
FILES = [
    r"D:\Andy\Inflam_NP\Predictive_models\Prob_NPASS\NPASS_test_pred_BiLSTM_rdkit.csv",
    r"D:\Andy\Inflam_NP\Predictive_models\Prob_NPASS\NPASS_test_pred_BiLSTM_ecfp.csv",
    r"D:\Andy\Inflam_NP\Predictive_models\Prob_NPASS\NPASS_test_pred_BiLSTM_maccs.csv",
    r"D:\Andy\Inflam_NP\Predictive_models\Prob_NPASS\NPASS_test_pred_XGB_rdkit.csv",
    r"D:\Andy\Inflam_NP\Predictive_models\Prob_NPASS\NPASS_test_pred_XGB_ecfp.csv",
    r"D:\Andy\Inflam_NP\Predictive_models\Prob_NPASS\NPASS_test_pred_XGB_maccs.csv",
]
KEY = "canonical_smiles"
OUTPUT_REPORT = "consensus_stats_report_filtered.csv"
# ====================


def infer_model_name(path: str) -> str:
  base = os.path.basename(path)
  base = re.sub(r"\.csv$", "", base)
  return base.replace("NPASS_test_pred_", "")


# 1. Đọc dữ liệu từ các file
model_dfs = []
model_names = []

for f in FILES:
  model = infer_model_name(f)
  model_names.append(model)

  df = pd.read_csv(f)
  prob_col = "y_pro_average" if "y_pro_average" in df.columns else "prob"

  df = df[[KEY, "y_pred", prob_col]].copy()
  df["y_pred"] = (
      pd.to_numeric(df["y_pred"], errors="coerce").fillna(0).astype(int)
  )
  df.rename(
      columns={"y_pred": f"pred_{model}", prob_col: f"prob_{model}"},
      inplace=True,
  )
  model_dfs.append(df)

# 2. Gộp dữ liệu chung
df_all = reduce(lambda l, r: pd.merge(l, r, on=KEY, how="inner"), model_dfs)
pred_cols = [f"pred_{m}" for m in model_names]
prob_cols = [f"prob_{m}" for m in model_names]

df_all["n_active"] = df_all[pred_cols].sum(axis=1).astype(int)
df_all["prob_avg"] = df_all[prob_cols].mean(axis=1)

# Lấy thêm các chỉ số lý tính từ file số 1 (nếu có)
df1 = pd.read_csv(FILES[0])
if "sa_score" in df1.columns:
  df_all = pd.merge(
      df_all,
      df1[[KEY, "sa_score", "lipinski_rule_of_five_violations"]],
      on=KEY,
      how="inner",
  )

# 3. LỌC BỎ 0/6 và 1/6 (Chỉ giữ lại các mẫu có n_active >= 2)
df_filtered = df_all[df_all["n_active"] >= 2].copy()

# 4. Tính Mean và SD theo từng mức độ đồng thuận (từ 6 xuống 2)
stats_summary = (
    df_filtered.groupby("n_active")
    .agg(
        n_compounds=("canonical_smiles", "count"),
        conf_mean=("prob_avg", "mean"),
        conf_std=("prob_avg", "std"),
        sa_mean=(
            ("sa_score", "mean")
            if "sa_score" in df_filtered.columns
            else ("n_active", "count")
        ),
        sa_std=(
            ("sa_score", "std")
            if "sa_score" in df_filtered.columns
            else ("n_active", "count")
        ),
    )
    .reset_index()
    .sort_values(by="n_active", ascending=False)
)

# 5. Định dạng dạng "mean ± SD" với 3 chữ số thập phân
n_models = len(model_names)
stats_summary["Consensus_Level"] = stats_summary["n_active"].apply(
    lambda x: f"{x}/{n_models}"
)

stats_summary["Prediction_Confidence (Mean ± SD)"] = stats_summary.apply(
    lambda row: f"{row['conf_mean']:.3f} ± {row['conf_std']:.3f}"
    if pd.notnull(row["conf_std"])
    else f"{row['conf_mean']:.3f} ± 0.000",
    axis=1,
)

if "sa_score" in df_filtered.columns:
  stats_summary["SA_Score (Mean ± SD)"] = stats_summary.apply(
      lambda row: f"{row['sa_mean']:.3f} ± {row['sa_std']:.3f}"
      if pd.notnull(row["sa_std"])
      else f"{row['sa_mean']:.3f} ± 0.000",
      axis=1,
  )
  final_report = stats_summary[[
      "Consensus_Level",
      "n_compounds",
      "Prediction_Confidence (Mean ± SD)",
      "SA_Score (Mean ± SD)",
  ]]
else:
  final_report = stats_summary[[
      "Consensus_Level",
      "n_compounds",
      "Prediction_Confidence (Mean ± SD)",
  ]]

# 6. Xuất ra file CSV sạch sẽ
final_report.to_csv(OUTPUT_REPORT, index=False)

print(f"\n Đã lưu báo cáo đã lọc (bỏ 0/6, 1/6) thành công tại: {OUTPUT_REPORT}")
print(final_report.to_string(index=False))