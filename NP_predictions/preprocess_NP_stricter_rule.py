# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from tqdm import tqdm

# ========= CONFIG =========
INPUT_CSV          = "NPASSv2.0.csv"
OUTPUT_FILTER_CSV  = "NPASSv2.0_filtered.csv"

REQ_COLS = ["canonical_smiles"]

# ==========================

# ---- Synthetic Accessibility Score (sascorer) ----
# Tải file sascorer.py từ RDKit contrib (https://github.com/rdkit/rdkit/tree/master/Contrib/SA_Score)
import sascorer  

def calc_sa_score(mol):
    try:
        return sascorer.calculateScore(mol)
    except Exception:
        return np.nan

# ---- PAINS + Brenk filters ----
def build_filter_catalog():
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog(params)

filter_catalog = build_filter_catalog()

def passes_filters(mol):
    if mol is None:
        return False
    if filter_catalog.HasMatch(mol):
        return False
    return True

# ---- Core drug-like criteria ----
def passes_druglike_criteria(mol):
    if mol is None:
        return False

    mw   = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd  = rdMolDescriptors.CalcNumHBD(mol)
    hba  = rdMolDescriptors.CalcNumHBA(mol)
    rb   = rdMolDescriptors.CalcNumRotatableBonds(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    qed  = QED.qed(mol)
    sa   = calc_sa_score(mol)

    if not (200 <= mw <= 600):
        return False
    if not (-2 <= logp <= 5):
        return False
    if hbd > 5 or hba > 10:
        return False
    if rb > 15 or tpsa > 150:
        return False
    if qed < 0.5:  # có thể chỉnh thành 0.4
        return False
    if sa > 2:
        return False
    if not passes_filters(mol):
        return False
    return True

# ---- Subclass filtering ----
def subclass_allowed(row):
    for col in ["np_classifier_is_glycoside", "np_classifier_is_peptide", "np_classifier_is_macrocycle"]:
        if col in row.index and str(row[col]).lower() in ["true", "1", "t", "yes", "y"]:
            return False
    return True

# ========== MAIN ==========
def main():
    df = pd.read_csv(INPUT_CSV)
    n0 = len(df)

    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {missing}")

    mols = [Chem.MolFromSmiles(str(s)) if pd.notna(s) else None 
            for s in tqdm(df["canonical_smiles"], desc="Parsing SMILES")]

    # Apply filters
    keep = []
    for idx, mol in tqdm(list(enumerate(mols)), desc="Filtering"):
        if mol is None:
            keep.append(False)
            continue
        if not subclass_allowed(df.loc[idx]):
            keep.append(False)
            continue
        if passes_druglike_criteria(mol):
            keep.append(True)
        else:
            keep.append(False)

    df_filt = df[keep].reset_index(drop=True)
    df_filt.to_csv(OUTPUT_FILTER_CSV, index=False)

    print("== Filter summary ==")
    print(f"Input rows         : {n0}")
    print(f"Kept rows          : {len(df_filt)}")
    print(f"Saved              : {OUTPUT_FILTER_CSV}")

if __name__ == "__main__":
    main()
