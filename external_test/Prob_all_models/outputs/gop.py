import pandas as pd
from functools import reduce

files = {
    "XGB_ecfp":  "InFlam_external_neg3times_test_prob_ecfp_mean_XGB.csv",
    "XGB_maccs": "InFlam_external_neg3times_test_prob_maccs_mean_XGB.csv",
    "XGB_rdkit": "InFlam_external_neg3times_test_prob_rdkit_mean_XGB.csv",
    "BiLSTM_ecfp":  "InFlam_external_neg3times_test_prob_ecfp_mean_BiLSTM.csv",
    "BiLSTM_maccs": "InFlam_external_neg3times_test_prob_maccs_mean_BiLSTM.csv",
    "BiLSTM_rdkit": "InFlam_external_neg3times_test_prob_rdkit_mean_BiLSTM.csv",
}

KEY_COL = "Compound name"
YTRUE_COL = "y_true"
YPROB_COL = "y_prob"

# --- 1) Đọc file đầu tiên làm "base" để giữ y_true ---
first_model = next(iter(files.keys()))
base_path = files[first_model]

base = pd.read_csv(base_path)
base[YTRUE_COL] = pd.to_numeric(base[YTRUE_COL], errors="coerce")
base[YPROB_COL] = pd.to_numeric(base[YPROB_COL], errors="coerce")

base = base[base[YTRUE_COL] == 1].copy()
base = base[[KEY_COL, YTRUE_COL, YPROB_COL]].dropna(subset=[KEY_COL]).drop_duplicates(KEY_COL)
base = base.rename(columns={YPROB_COL: f"{YPROB_COL}_{first_model}"})

# --- 2) Đọc các file còn lại: CHỈ giữ Compound name + y_prob_<model> (không mang y_true) ---
others = []
for model_name, path in list(files.items())[1:]:
    df = pd.read_csv(path)
    df[YTRUE_COL] = pd.to_numeric(df[YTRUE_COL], errors="coerce")
    df[YPROB_COL] = pd.to_numeric(df[YPROB_COL], errors="coerce")

    df = df[df[YTRUE_COL] == 1].copy()
    df = df[[KEY_COL, YPROB_COL]].dropna(subset=[KEY_COL]).drop_duplicates(KEY_COL)
    df = df.rename(columns={YPROB_COL: f"{YPROB_COL}_{model_name}"})

    others.append(df)

# --- 3) Merge: base (có y_true) + các bảng prob ---
merged = reduce(lambda l, r: pd.merge(l, r, on=KEY_COL, how="inner"), [base] + others)

# --- 4) Sắp xếp cột ---
prob_cols = [f"{YPROB_COL}_{m}" for m in files.keys()]
merged = merged[[KEY_COL, YTRUE_COL] + prob_cols].sort_values(KEY_COL).reset_index(drop=True)

# --- 4.5) Làm tròn ---
merged[prob_cols] = merged[prob_cols].round(3)

# --- 5) Xuất file ---
out = "merged_ytrue1.csv"
merged.to_csv(out, index=False, float_format="%.3f")
print("Saved:", out, "| rows:", len(merged), "| cols:", len(merged.columns))