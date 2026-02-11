import pandas as pd
from pathlib import Path

# =========================
# Config
# =========================
IN_DIR = Path("outputs")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "XGB_ECFP":    "InFlam_external_neg3times_test_prob_ecfp_mean_XGB.csv",
    "XGB_MACCS":   "InFlam_external_neg3times_test_prob_maccs_mean_XGB.csv",
    "XGB_RDKIT":   "InFlam_external_neg3times_test_prob_rdkit_mean_XGB.csv",
    "BiLSTM_ECFP":  "InFlam_external_neg3times_test_prob_ecfp_mean_BiLSTM.csv",
    "BiLSTM_MACCS": "InFlam_external_neg3times_test_prob_maccs_mean_BiLSTM.csv",
    "BiLSTM_RDKIT": "InFlam_external_neg3times_test_prob_rdkit_mean_BiLSTM.csv",
}

THRESH = 0.5
ROUND_N = 3

NAME_COL_CANDIDATES = [
    "Compound name", "compound name",
    "Compound_name", "compound_name",
    "Name", "name"
]

# =========================
# Helpers
# =========================
def find_name_col(df: pd.DataFrame) -> str:
    for c in NAME_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"Không tìm thấy cột compound name. Columns: {list(df.columns)}")

def load_model_df(model: str, filename: str) -> pd.DataFrame:
    path = IN_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    df = pd.read_csv(path)

    if "y_true" not in df.columns or "y_prob" not in df.columns:
        raise ValueError(f"{path.name} thiếu cột y_true hoặc y_prob. Columns: {list(df.columns)}")

    name_col = find_name_col(df)

    out = df[[name_col, "y_true", "y_prob"]].copy()
    out = out.rename(columns={name_col: "Compound name"})
    out["y_true"] = pd.to_numeric(out["y_true"], errors="raise")
    out["y_prob"] = pd.to_numeric(out["y_prob"], errors="raise")
    out["Model"] = model
    return out

# =========================
# Main
# =========================
if __name__ == "__main__":

    # 1) long format: stack 6 models
    long_df = pd.concat(
        [load_model_df(m, f) for m, f in FILES.items()],
        ignore_index=True
    )

    # 2) filter y_true == 1
    pos_df = long_df[long_df["y_true"] == 1].copy()

    # 3) pivot to wide: each compound 1 row, each model 1 column
    wide = pos_df.pivot_table(
        index="Compound name",
        columns="Model",
        values="y_prob",
        aggfunc="first"  # mỗi compound-model nên chỉ có 1 giá trị
    )

    # keep only rows with all 6 models
    wide = wide.dropna()

    # 4) disagreement filter at 0.5
    disagree = wide[(wide.min(axis=1) < THRESH) & (wide.max(axis=1) >= THRESH)].copy()

    # 5) add mean + predicted label (based on mean)
    disagree["Mean_6Models"] = disagree.mean(axis=1)
    disagree["Pred_label"] = disagree["Mean_6Models"].apply(lambda x: "Active" if x >= THRESH else "Inactive")

    # 6) round y_prob columns + mean (3 decimals)
    model_cols = list(FILES.keys())
    disagree[model_cols] = disagree[model_cols].round(ROUND_N)
    disagree["Mean_6Models"] = disagree["Mean_6Models"].round(ROUND_N)

    # 7) export
    out_df = disagree.reset_index()  # bring Compound name back as a column

    # optional: order columns nicely
    out_df = out_df[["Compound name"] + model_cols + ["Mean_6Models", "Pred_label"]]

    out_path = OUT_DIR / "disagreement_ytrue1_threshold0p5_wide_mean_label.csv"
    out_df.to_csv(out_path, index=False)

    print(f"[OK] Total y_true=1 compounds with all 6 models: {wide.shape[0]}")
    print(f"[OK] Disagreement compounds: {out_df.shape[0]}")
    print(f"[OK] Output saved to: {out_path}")
