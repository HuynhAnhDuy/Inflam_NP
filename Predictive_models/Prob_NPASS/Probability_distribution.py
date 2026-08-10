import os
import re
import numpy as np
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

OUT_SUMMARY = "NPASS_prob_summary_supporters_k2_to_k6.csv"
OUT_FIG = "NPASS_supportprob_boxplot_k2_to_k6.svg"

MIN_SUPPORT_MODELS = 2
# ====================


def infer_model_name(path: str) -> str:
    base = os.path.basename(path)
    base = re.sub(r"\.csv$", "", base)
    return base.replace("NPASS_test_pred_", "")


# ====== 1) Load + merge all models ======
model_dfs = []
model_names = []

for f in FILES:
    model = infer_model_name(f)
    model_names.append(model)

    df = pd.read_csv(f)
    need = {KEY, "y_pred", "y_pro_average"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"File {f} missing columns: {miss}")

    df = df[[KEY, "y_pred", "y_pro_average"]].copy()
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce").fillna(0).astype(int)
    df["y_pro_average"] = pd.to_numeric(df["y_pro_average"], errors="coerce")

    df.rename(columns={
        "y_pred": f"pred_{model}",
        "y_pro_average": f"prob_{model}",
    }, inplace=True)

    model_dfs.append(df)

df_all = reduce(lambda l, r: pd.merge(l, r, on=KEY, how="inner"), model_dfs)

pred_cols = [f"pred_{m}" for m in model_names]
prob_cols = [f"prob_{m}" for m in model_names]
n_models = len(model_names)

# ====== 2) Consensus count (k/6) ======
df_all["n_active"] = df_all[pred_cols].sum(axis=1).astype(int)

# ====== 3) Compute probability ONLY among supporting models ======
pred_mat = df_all[pred_cols].to_numpy(dtype=float)   # (N,6) 0/1
prob_mat = df_all[prob_cols].to_numpy(dtype=float)   # (N,6)

masked_prob = np.where(pred_mat == 1, prob_mat, np.nan)

df_all["prob_support_avg"] = np.nanmean(masked_prob, axis=1)
df_all["prob_support_median"] = np.nanmedian(masked_prob, axis=1)
df_all["prob_support_min"] = np.nanmin(masked_prob, axis=1)
df_all["prob_support_max"] = np.nanmax(masked_prob, axis=1)

# Lấy thêm thông tin sa_score từ file đầu tiên nếu có
df1 = pd.read_csv(FILES[0])
if "sa_score" in df1.columns:
    df_all = pd.merge(
        df_all,
        df1[[KEY, "sa_score"]],
        on=KEY,
        how="inner"
    )

# ====== 4) Focus on predicted actives only: keep k=2..6 ======
keep_levels = list(range(MIN_SUPPORT_MODELS, n_models + 1))
df_act = df_all[df_all["n_active"].isin(keep_levels)].copy()
df_act = df_act.dropna(subset=["prob_support_avg"])

# ====== 5) Summary stats by exact agreement level (k) for actives ======
agg_dict = {
    "n_compounds": (KEY, "count"),
    "conf_mean": ("prob_support_avg", "mean"),
    "conf_std": ("prob_support_avg", "std"),
}

if "sa_score" in df_act.columns:
    agg_dict["sa_mean"] = ("sa_score", "mean")
    agg_dict["sa_std"] = ("sa_score", "std")

summary = (
    df_act
    .groupby("n_active")
    .agg(**agg_dict)
    .reindex(keep_levels)
    .reset_index()
)

summary["Consensus_Level"] = summary["n_active"].astype(str) + f"/{n_models}"

# Định dạng cột Prediction_Confidence (Mean ± SD)
summary["Prediction_Confidence (Mean ± SD)"] = summary.apply(
    lambda row: f"{row['conf_mean']:.3f} ± {row['conf_std']:.3f}"
    if pd.notnull(row["conf_std"])
    else f"{row['conf_mean']:.3f} ± 0.000",
    axis=1,
)

# Định dạng cột SA_Score (Mean ± SD)
if "sa_score" in df_act.columns:
    summary["SA_Score (Mean ± SD)"] = summary.apply(
        lambda row: f"{row['sa_mean']:.3f} ± {row['sa_std']:.3f}"
        if pd.notnull(row["sa_std"])
        else f"{row['sa_mean']:.3f} ± 0.000",
        axis=1,
    )
else:
    summary["SA_Score (Mean ± SD)"] = "N/A"

# Chỉ chọn đúng 4 cột yêu cầu và SẮP XẾP GIẢM DẦN theo n_active (6/6 xuống 2/6)
final_summary = summary[[
    "Consensus_Level",
    "n_compounds",
    "Prediction_Confidence (Mean ± SD)",
    "SA_Score (Mean ± SD)",
    "n_active" # Giữ tạm cột n_active để sort chính xác
]].sort_values(by="n_active", ascending=False).drop(columns=["n_active"])

final_summary.to_csv(OUT_SUMMARY, index=False)
print("Saved summary:", OUT_SUMMARY)
print(final_summary.to_string(index=False))

# ====== 6) Boxplot (top) + "Consensus level" (middle) + table (bottom) ======
order = sorted(keep_levels, reverse=True)  # 6 -> 2
labels = [f"{k}/{n_models}" for k in order]
data = [df_act.loc[df_act["n_active"] == k, "prob_support_avg"].dropna().values for k in order]

fig = plt.figure(figsize=(10, 7))

# 3 rows: boxplot / xlabel / table
gs = fig.add_gridspec(
    nrows=3, ncols=1,
    height_ratios=[4.6, 0.35, 1.25],
    hspace=0.05
)

ax = fig.add_subplot(gs[0, 0])     # boxplot
ax_xlab = fig.add_subplot(gs[1, 0])  # just for "Consensus level"
ax_tbl = fig.add_subplot(gs[2, 0])   # table

# --- Boxplot ---
bp = ax.boxplot(
    data,
    labels=labels,
    showfliers=False,
    showmeans=True,
    medianprops=dict(color="red", linewidth=3.0, zorder=5),
    meanprops=dict(
        marker="^",
        markerfacecolor="blue",
        markeredgecolor="black",
        markersize=7
    ),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
    boxprops=dict(linewidth=1.2),
)

ax.set_ylabel("Predicted probability (positive-voting)",
              fontsize=12, fontweight='bold', fontstyle='italic', family='sans-serif')

ax.tick_params(axis="x", labelbottom=False)

# --- Legend giải thích mean/median ---
handles = [
    Line2D([0], [0],
           marker="^", linestyle="None", markersize=7,
           markerfacecolor="blue", markeredgecolor="black",
           label="Mean"),
    Line2D([0], [0],
           color="red", linewidth=3.0,
           label="Median"),
]
ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.1, 0.98), frameon=True)

# --- Middle axis: only the xlabel text ---
ax_xlab.axis("off")
ax_xlab.text(
    0.5, 0.5, "Consensus level",
    ha="center", va="center",
    fontsize=12, fontweight="bold", fontstyle="italic"
)

# --- Build table values: n, mean, median ---
table_rows = ["# compounds", "Mean prob.", "Median prob."]
cell_text = []

for row in table_rows:
    row_vals = []
    for k in order:
        vals = df_act.loc[df_act["n_active"] == k, "prob_support_avg"].dropna()
        if len(vals) == 0:
            row_vals.append("—")
            continue

        if row == "# compounds":
            row_vals.append(str(int(len(vals))))
        elif row == "Mean prob.":
            row_vals.append(f"{float(vals.mean()):.3f}")
        elif row == "Median prob.":
            row_vals.append(f"{float(vals.median()):.3f}")
    cell_text.append(row_vals)

# --- Table axis ---
ax_tbl.axis("off")
tbl = ax_tbl.table(
    cellText=cell_text,
    rowLabels=table_rows,
    colLabels=labels,
    cellLoc="center",
    rowLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)

for cell in tbl.get_celld().values():
    cell.get_text().set_fontfamily("sans-serif")
    cell.get_text().set_fontstyle("normal")
    cell.get_text().set_fontweight("normal")

tbl.scale(1.0, 1.2)

cells = tbl.get_celld()

# Header hàng (row labels)
for r in range(len(table_rows)):
    cells[(r + 1, -1)].get_text().set_fontweight("bold")

# Header cột (col labels)
for c in range(len(labels)):
    cells[(0, c)].get_text().set_fontweight("bold")

plt.savefig(OUT_FIG, format="svg", bbox_inches="tight")
plt.close(fig)
print("Saved figure:", OUT_FIG)