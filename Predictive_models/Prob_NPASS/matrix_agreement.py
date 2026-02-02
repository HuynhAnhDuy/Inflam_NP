import os
import re
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
KEY = "canonical_smiles"
OUT_SUMMARY_GE = "NPASS_agreement_summary_ge_k.csv"
# ====================


def infer_model_name(path: str) -> str:
    base = os.path.basename(path)
    base = re.sub(r"\.csv$", "", base)
    return base.replace("NPASS_test_pred_", "")


# ====== 1) Load only KEY + y_pred for each model ======
model_dfs = []
model_names = []

for f in FILES:
    model = infer_model_name(f)
    model_names.append(model)

    df = pd.read_csv(f)
    if KEY not in df.columns or "y_pred" not in df.columns:
        raise ValueError(f"Missing required columns in {f}. Need: {KEY}, y_pred")

    df = df[[KEY, "y_pred"]].copy()
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce").fillna(0).astype(int)
    df.rename(columns={"y_pred": f"pred_{model}"}, inplace=True)

    model_dfs.append(df)

# ====== 2) Merge predictions (inner assumes the same compounds across files) ======
df_all = reduce(lambda l, r: pd.merge(l, r, on=KEY, how="inner"), model_dfs)

pred_cols = [f"pred_{m}" for m in model_names]
n_models = len(model_names)

# ====== 3) Count #models predicting active per compound ======
df_all["n_active"] = df_all[pred_cols].sum(axis=1).astype(int)

# ====== 4) Count n_compounds for each k (0..6), then compute cumulative >=k ======
counts = (
    df_all.groupby("n_active")
    .size()
    .reindex(range(0, n_models + 1), fill_value=0)   # ensure rows 0..6 exist
    .sort_index(ascending=False)                      # 6,5,4,...,0
)

# counts: index = k, value = n_compounds with exactly k models predicting active
summary = pd.DataFrame({
    "k_over_6": [f"{k}/{n_models}" for k in counts.index],
    "n_active": counts.index,
    "n_compounds": counts.values,
})

# cumulative ">=k": from top down (6->0)
summary["n_compounds_ge_k"] = summary["n_compounds"].cumsum()

# optional: drop n_active if you only want k/6 display
summary = summary[["k_over_6", "n_compounds", "n_compounds_ge_k"]]

summary.to_csv(OUT_SUMMARY_GE, index=False)

print("Saved:", OUT_SUMMARY_GE)
print(summary.to_string(index=False))
