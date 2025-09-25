# -*- coding: utf-8 -*-
import pandas as pd

# ======= CONFIG =======
INFLAM_CSV   = "/home/andy/andy/Inflam_NP/Scaffold_identify/3.InFlamNat_preprocess.csv"
EXACT_CSV    = "/home/andy/andy/Inflam_NP/Scaffold_identify/NP_candidate/Coconut_NP_exact.csv"
SIMILAR_CSV  = "/home/andy/andy/Inflam_NP/Scaffold_identify/NP_candidate/Coconut_NP_similar.csv"

# Tên cột SMILES có thể gặp (ưu tiên đúng canonical_smiles nếu có)
SMILES_COL_CANDIDATES = ["canonical_smiles", "canonical_SMILES", "SMILES", "smiles"]

# Output (đÃ lọc bỏ những dòng có canonical_smiles xuất hiện trong InFlam.csv)
EXACT_OUT_MOD    = "/home/andy/andy/Inflam_NP/Scaffold_identify/NP_candidate/NP_candidates_exact_modified.csv"
SIMILAR_OUT_MOD  = "/home/andy/andy/Inflam_NP/Scaffold_identify/NP_candidate/NP_candidates_similar_modified.csv"
# ======================


def pick_smiles_col(df: pd.DataFrame) -> str:
    for c in SMILES_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"Không tìm thấy cột SMILES. Thử một trong: {SMILES_COL_CANDIDATES}")

def normalize_smiles_series(s: pd.Series) -> pd.Series:
    # Chuẩn hoá đơn giản để so sánh chuỗi: strip + lower (KHÔNG tái-canonical bằng RDKit)
    return s.astype(str).str.strip().str.lower()

def load_inflam_smiles(path: str) -> set:
    df = pd.read_csv(path)
    smi_col = pick_smiles_col(df)
    inflam_smiles = normalize_smiles_series(df[smi_col]).dropna()
    return set(inflam_smiles.tolist())

def filter_np_against_inflam(np_path: str, inflam_set: set, out_path: str):
    df = pd.read_csv(np_path)
    smi_col = pick_smiles_col(df)

    # Tạo cột tạm để so sánh
    norm_col = "__norm_smiles__"
    df[norm_col] = normalize_smiles_series(df[smi_col])

    # Đánh dấu trùng với InFlam
    df["exists_in_InFlam"] = df[norm_col].isin(inflam_set)

    # Lọc bỏ các dòng trùng (giữ lại chỉ các dòng KHÔNG trùng với InFlam)
    df_mod = df[~df["exists_in_InFlam"]].drop(columns=[norm_col])

    # Ghi file modified
    df_mod.to_csv(out_path, index=False)

    # Log nhanh
    total = len(df)
    dup   = int(df["exists_in_InFlam"].sum())
    kept  = len(df_mod)
    print(f"[{np_path}] Tổng: {total} | Trùng với InFlam: {dup} | Giữ lại: {kept} → Saved: {out_path}")

def main():
    print("== Kiểm tra trùng canonical_smiles với InFlam và tạo file modified ==")
    inflam_set = load_inflam_smiles(INFLAM_CSV)
    print(f"Loaded InFlam unique SMILES: {len(inflam_set)}")

    filter_np_against_inflam(EXACT_CSV,   inflam_set, EXACT_OUT_MOD)
    filter_np_against_inflam(SIMILAR_CSV, inflam_set, SIMILAR_OUT_MOD)

if __name__ == "__main__":
    main()
