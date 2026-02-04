# -*- coding: utf-8 -*-
"""
Count + list compounds in File B that match target scaffolds in File A.

Inputs:
  - File A: CSV with column "scaffold" (target scaffolds, e.g., 3 scaffolds)
  - File B: CSV with column "canonical_smiles" (+ optional compound id column)

Outputs:
  - scaffold_counts.csv
  - scaffold_hits_long.csv
  - scaffold_hits_wide.xlsx (each scaffold = 1 sheet)
  - B_with_scaffold.csv (optional, for debug)
"""

import os
import re
import sys
from typing import Optional

import pandas as pd

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize


# =========================
# 1) Your scaffold functions
# =========================
def _standardize_mol(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Chuẩn hoá phân tử trước khi lấy scaffold"""
    if mol is None:
        return None
    try:
        params = rdMolStandardize.CleanupParameters()
        mol = rdMolStandardize.Cleanup(mol, params)
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)     # giữ mảnh lớn nhất
        mol = rdMolStandardize.Uncharger().uncharge(mol)                # trung hoá điện tích
        mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)   # canonical tautomer
        return mol
    except Exception:
        return None


def get_scaffold(smiles: str) -> Optional[str]:
    """Trích xuất Murcko scaffold đã chuẩn hoá từ SMILES"""
    if smiles is None:
        return None
    smiles = str(smiles).strip()
    if smiles == "" or smiles.lower() in {"nan", "none"}:
        return None

    mol = Chem.MolFromSmiles(smiles)
    mol = _standardize_mol(mol)
    if mol is None:
        return None
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        if core is None or core.GetNumAtoms() == 0:
            return None
        return Chem.MolToSmiles(core, isomericSmiles=False,
                                kekuleSmiles=False, canonical=True)
    except Exception:
        return None


# =========================
# 2) Helpers
# =========================
def safe_sheet_name(name: str, max_len: int = 31) -> str:
    """Excel sheet name max 31 chars and cannot contain: : \ / ? * [ ]"""
    name = str(name)
    name = re.sub(r"[:\\/?*\[\]]", "_", name)
    name = name.strip()
    if len(name) == 0:
        name = "sheet"
    return name[:max_len]


# =========================
# 3) Main pipeline
# =========================
def main():
    # ---- EDIT PATHS HERE ----
    FILE_A = r"scaffold_shap_summary.csv"
    FILE_B = r"InFlam_full.csv"

    COL_SCAFF_A = "scaffold"
    COL_SMILES_B = "canonical_smiles"

    # Optional: if B has an ID column, set it here. If None, use row_index
    ID_COL_B = None   # e.g., "compound_id"

    OUT_COUNTS = "scaffold_counts.csv"
    OUT_HITS_LONG = "scaffold_hits_long.csv"
    OUT_HITS_WIDE_XLSX = "scaffold_hits_wide.xlsx"
    OUT_B_WITH_SCAFF = "B_with_scaffold.csv"  # set to None if you don't want

    # ---- Load A ----
    if not os.path.exists(FILE_A):
        raise FileNotFoundError(f"Not found: {FILE_A}")
    dfA = pd.read_csv(FILE_A)

    if COL_SCAFF_A not in dfA.columns:
        raise ValueError(f"File A missing column '{COL_SCAFF_A}'. Found: {list(dfA.columns)}")

    target_scaffolds = (
        dfA[COL_SCAFF_A]
        .dropna()
        .astype(str)
        .str.strip()
    )
    target_scaffolds = target_scaffolds[target_scaffolds != ""].unique().tolist()
    if len(target_scaffolds) == 0:
        raise ValueError("No scaffolds found in File A after cleaning.")

    target_set = set(target_scaffolds)

    print(f"[INFO] Target scaffolds from A: {len(target_scaffolds)}")
    for i, scf in enumerate(target_scaffolds, 1):
        print(f"  {i}. {scf}")

    # ---- Load B ----
    if not os.path.exists(FILE_B):
        raise FileNotFoundError(f"Not found: {FILE_B}")
    dfB = pd.read_csv(FILE_B)

    if COL_SMILES_B not in dfB.columns:
        raise ValueError(f"File B missing column '{COL_SMILES_B}'. Found: {list(dfB.columns)}")

    # Create an ID column for reporting
    if ID_COL_B and ID_COL_B in dfB.columns:
        dfB["_compound_id"] = dfB[ID_COL_B].astype(str)
    else:
        dfB["_compound_id"] = dfB.index.astype(int)  # stable row index

    # ---- Compute scaffold for B ----
    print("[INFO] Computing scaffold for File B...")
    dfB["scaffold_calc"] = dfB[COL_SMILES_B].apply(get_scaffold)

    # ---- Filter hits in target scaffolds ----
    hits = dfB[dfB["scaffold_calc"].isin(target_set)].copy()

    # ---- Counts table (include scaffold with 0 hits) ----
    counts = hits["scaffold_calc"].value_counts().to_dict()

    df_counts = pd.DataFrame({
        "scaffold": target_scaffolds,
        "count_in_B": [int(counts.get(s, 0)) for s in target_scaffolds]
    }).sort_values("count_in_B", ascending=False)

    df_counts.to_csv(OUT_COUNTS, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved counts: {OUT_COUNTS}")

    # ---- Long hits table: list each compound explicitly ----
    # Keep extra columns if you want (uncomment as needed)
    keep_cols = ["_compound_id", COL_SMILES_B, "scaffold_calc"]
    # Add more columns from B if present:
    # for col in ["name", "label", "source"]:
    #     if col in dfB.columns:
    #         keep_cols.append(col)

    hits_long = hits[keep_cols].rename(columns={
        "_compound_id": "compound_id",
        COL_SMILES_B: "canonical_smiles",
        "scaffold_calc": "scaffold"
    }).sort_values(["scaffold", "compound_id"])

    hits_long.to_csv(OUT_HITS_LONG, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved hit list (long): {OUT_HITS_LONG}")

    # ---- Wide Excel: each scaffold a sheet with its compounds ----
    with pd.ExcelWriter(OUT_HITS_WIDE_XLSX, engine="openpyxl") as writer:
        # Write summary sheet
        df_counts.to_excel(writer, sheet_name="summary_counts", index=False)

        # Write one sheet per scaffold
        for scf in target_scaffolds:
            sub = hits_long[hits_long["scaffold"] == scf].copy()
            sheet = safe_sheet_name(scf)
            # If no hits, still create an empty sheet with headers
            sub.to_excel(writer, sheet_name=sheet, index=False)

    print(f"[OK] Saved hit list (wide Excel): {OUT_HITS_WIDE_XLSX}")

    # ---- Optional: save B with computed scaffold ----
    if OUT_B_WITH_SCAFF:
        dfB.to_csv(OUT_B_WITH_SCAFF, index=False, encoding="utf-8-sig")
        print(f"[OK] Saved File B with computed scaffold: {OUT_B_WITH_SCAFF}")

    # ---- Summary to screen ----
    n_total = len(dfB)
    n_none = dfB["scaffold_calc"].isna().sum()
    n_valid = n_total - n_none
    n_hits = len(hits)

    print("\n========== SUMMARY ==========")
    print(f"Total rows in B: {n_total}")
    print(f"Valid scaffold computed: {n_valid}")
    print(f"Scaffold None/invalid: {n_none}")
    print(f"Hits in target scaffolds: {n_hits}")
    print("\nCounts per target scaffold:")
    print(df_counts.to_string(index=False))

    # Also show a quick preview of hits
    if n_hits > 0:
        print("\nPreview hits (first 20 rows):")
        print(hits_long.head(20).to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise
