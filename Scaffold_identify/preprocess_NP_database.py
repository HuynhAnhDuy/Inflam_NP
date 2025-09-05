# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# ========= CONFIG =========
INPUT_CSV          = "coconut_csv-09-2025.csv"
OUTPUT_FILTER_CSV  = "coconut_csv-09-2025_clean.csv"

MAX_LIPINSKI_VIOL  = 1      # giữ: số vi phạm RO5 <= 1
MIN_NP_LIKENESS    = 0.0    # giữ: np_likeness >= 0.0
# ==========================

REQ_COLS = [
    "lipinski_rule_of_five_violations",
    "np_likeness",
    "np_classifier_is_glycoside",
]

def to_bool_series(x: pd.Series) -> pd.Series:
    if x.dtype == bool:
        return x
    x = x.astype(str).str.strip().str.lower()
    return x.isin(["true","1","t","yes","y"])

def main():
    df = pd.read_csv(INPUT_CSV)
    n0 = len(df)

    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {missing}")

    # Ép kiểu an toàn
    df["lipinski_rule_of_five_violations"] = pd.to_numeric(
        df["lipinski_rule_of_five_violations"], errors="coerce"
    )
    df["np_likeness"] = pd.to_numeric(df["np_likeness"], errors="coerce")
    df["np_classifier_is_glycoside"] = to_bool_series(df["np_classifier_is_glycoside"])

    # Lọc: ≤ 1 vi phạm RO5, np_likeness ≥ 0.0, và glycoside == False
    mask = (
        (df["lipinski_rule_of_five_violations"] <= MAX_LIPINSKI_VIOL) &
        (df["np_likeness"] >= MIN_NP_LIKENESS) &
        (df["np_classifier_is_glycoside"] == False)   # giữ False
    )

    df_filt = df[mask].reset_index(drop=True)
    df_filt.to_csv(OUTPUT_FILTER_CSV, index=False)

    print("== Filter summary ==")
    print(f"Input rows         : {n0}")
    print(f"Kept rows          : {len(df_filt)}")
    print(f"Criteria           : RO5_viol <= {MAX_LIPINSKI_VIOL}, "
          f"np_likeness >= {MIN_NP_LIKENESS}, np_classifier_is_glycoside == False")
    print(f"Saved              : {OUTPUT_FILTER_CSV}")

if __name__ == "__main__":
    main()
