# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from tqdm import tqdm

# ========= CONFIG =========
INPUT_CSV          = "/home/andy/andy/Inflam_NP/NP_predictions/XGB_shap_compounds_safety.csv"
OUTPUT_FILTER_CSV  = "XGB_shap_compounds_LRo5_SA_2.csv"

MAX_LIPINSKI_VIOL  = 1       # giữ: số vi phạm RO5 <= 1
MAX_SA_SCORE       = 2.0     # SA score <= 5
# ==========================

REQ_COLS = ["canonical_smiles"]

# ========= IMPORT SASCORER =========
# tải từ https://github.com/rdkit/rdkit/tree/master/Contrib/SA_Score
import sascorer  

def calc_sa_score(mol):
    """Tính Synthetic Accessibility Score (1 = dễ tổng hợp, 10 = khó)"""
    try:
        return sascorer.calculateScore(mol)
    except Exception:
        return np.nan

# ========== UTILS ==========
def calc_ro5_violations(mol):
    """Tính số vi phạm Lipinski's Rule of Five"""
    mw   = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd  = rdMolDescriptors.CalcNumHBD(mol)
    hba  = rdMolDescriptors.CalcNumHBA(mol)

    viol = 0
    if mw > 500: viol += 1
    if logp > 5: viol += 1
    if hbd > 5:  viol += 1
    if hba > 10: viol += 1
    return viol

def to_bool_series(x: pd.Series) -> pd.Series:
    if x.dtype == bool:
        return x
    x = x.astype(str).str.strip().str.lower()
    return x.isin(["true","1","t","yes","y"])

# ========== MAIN ==========
def main():
    df = pd.read_csv(INPUT_CSV)
    n0 = len(df)

    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {missing}")

    # Chuẩn hóa molecule
    mols = [Chem.MolFromSmiles(str(s)) if pd.notna(s) else None 
            for s in tqdm(df["canonical_smiles"], desc="Parsing SMILES")]

    # Bổ sung cột nếu thiếu
    if "lipinski_rule_of_five_violations" not in df.columns:
        df["lipinski_rule_of_five_violations"] = [
            calc_ro5_violations(m) if m else np.nan 
            for m in tqdm(mols, desc="Tính RO5 violations")
        ]

    if "sa_score" not in df.columns:
        df["sa_score"] = [
            calc_sa_score(m) if m else np.nan
            for m in tqdm(mols, desc="Tính SA score")
        ]

    if "np_classifier_is_glycoside" in df.columns:
        df["np_classifier_is_glycoside"] = to_bool_series(df["np_classifier_is_glycoside"])
        use_glyco = True
    else:
        use_glyco = False

    # Ép kiểu an toàn
    df["lipinski_rule_of_five_violations"] = pd.to_numeric(
        df["lipinski_rule_of_five_violations"], errors="coerce"
    )
    df["sa_score"] = pd.to_numeric(df["sa_score"], errors="coerce")

    # Lọc dữ liệu (theo RO5, SA và glycoside nếu có)
    mask = (
        (df["lipinski_rule_of_five_violations"] <= MAX_LIPINSKI_VIOL) &
        (df["sa_score"] <= MAX_SA_SCORE)
    )
    if use_glyco:
        mask &= (df["np_classifier_is_glycoside"] == False)

    df_filt = df[mask].reset_index(drop=True)
    df_filt.to_csv(OUTPUT_FILTER_CSV, index=False)

    # === Summary ===
    print("== Filter summary ==")
    print(f"Input rows         : {n0}")
    print(f"Kept rows          : {len(df_filt)}")
    print(f"Criteria           : RO5_viol <= {MAX_LIPINSKI_VIOL}, SA_score <= {MAX_SA_SCORE}"
          + (", np_classifier_is_glycoside == False" if use_glyco else ""))
    print(f"Saved              : {OUTPUT_FILTER_CSV}")

if __name__ == "__main__":
    main()
