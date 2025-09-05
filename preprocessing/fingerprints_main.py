"""
This software is used for Advanced Pharmaceutical Analysis
Prof(Assist).Dr.Tarapong Srisongkram – Khon Kaen University
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors, AllChem, MACCSkeys
from rdkit.Chem.EState import Fingerprinter  # E-State
import os

# ------------------------- 1.  ECFP (Morgan) ------------------------- #
def calculate_ecfp(df, smiles_col, radius=10, nBits=4096):
    def get_ecfp(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return [None] * nBits
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
            return [int(x) for x in fp.ToBitString()]
        except Exception:
            return [None] * nBits
    ecfp_df = df[smiles_col].apply(get_ecfp).apply(pd.Series)
    ecfp_df.columns = [f"ECFP{i}" for i in range(nBits)]
    return ecfp_df

# ------------------------- 2.  RDKit path-based ---------------------- #
def calculate_rdkit(df, smiles_col, nBits=2048):
    def get_rdkit(smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return [None] * nBits
            fp = Chem.RDKFingerprint(mol)
            return [int(x) for x in fp.ToBitString()]
        except Exception:
            return [None] * nBits
    rdk_df = df[smiles_col].apply(get_rdkit).apply(pd.Series)
    rdk_df.columns = [f"RDKit{i}" for i in range(nBits)]
    return rdk_df

# ------------------------- 3.  MACCS keys (167) ---------------------- #
def calculate_maccs(df, smiles_col):
    def get_maccs(smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return [None] * 167
            fp = MACCSkeys.GenMACCSKeys(mol)
            return [int(x) for x in fp.ToBitString()]
        except Exception:
            return [None] * 167
    maccs_df = df[smiles_col].apply(get_maccs).apply(pd.Series)
    maccs_df.columns = [f"MACCS{i}" for i in range(167)]
    return maccs_df

# ------------------------- 4.  E-State (79 continuous) --------------- #
def calculate_estate(df, smiles_col):
    """79-dimensional E-State indices (continuous values, rounded to 3 dp)."""
    def get_estate(smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return [None] * 79
            values = Fingerprinter.FingerprintMol(mol)[0]
            return [round(v, 3) for v in values]
        except Exception:
            return [None] * 79
    est_df = df[smiles_col].apply(get_estate).apply(pd.Series)
    est_df.columns = [f"EState_{i+1}" for i in range(79)]
    return est_df

# ------------------------- 5.  Physicochemical descriptors ----------- #
def calculate_phychem(df, smiles_col):
    descriptor_funcs = {
        "MolWt": Descriptors.MolWt,
        "LogP": Descriptors.MolLogP,
        "NumHDonors": Descriptors.NumHDonors,
        "NumHAcceptors": Descriptors.NumHAcceptors,
        "TPSA": rdMolDescriptors.CalcTPSA,
        "NumRotatableBonds": Descriptors.NumRotatableBonds,
        "NumAromaticRings": Descriptors.NumAromaticRings,
        "NumSaturatedRings": rdMolDescriptors.CalcNumSaturatedRings,
        "NumHeteroatoms": rdMolDescriptors.CalcNumHeteroatoms,
        "RingCount": rdMolDescriptors.CalcNumRings,
        "HeavyAtomCount": rdMolDescriptors.CalcNumHeavyAtoms,
        "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings,
    }
    def get_desc(smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return [None] * len(descriptor_funcs)
            return [func(mol) for func in descriptor_funcs.values()]
        except Exception:
            return [None] * len(descriptor_funcs)
    desc_df = df[smiles_col].apply(get_desc).apply(pd.Series)
    desc_df.columns = list(descriptor_funcs.keys())
    return desc_df

# --------------------------------------------------------------------- #
# 6. Helper: add index & save
# --------------------------------------------------------------------- #
def add_index_and_save(df_fps, df_src_index, filename):
    df_fps.insert(0, "Index", df_src_index)
    df_fps.to_csv(filename, index=False)

# --------------------------------------------------------------------- #
# 7. Main processing function (NO Atom Pair)
# --------------------------------------------------------------------- #
def process_and_save_features(df, smiles_col, prefix):
    # ECFP
    add_index_and_save(calculate_ecfp(df, smiles_col), df.index, f"{prefix}_ecfp.csv")
    # RDKit
    add_index_and_save(calculate_rdkit(df, smiles_col), df.index, f"{prefix}_rdkit.csv")
    # MACCS
    add_index_and_save(calculate_maccs(df, smiles_col), df.index, f"{prefix}_maccs.csv")
    # E-State
    add_index_and_save(calculate_estate(df, smiles_col), df.index, f"{prefix}_estate.csv")
    # Physicochemical
    add_index_and_save(calculate_phychem(df, smiles_col), df.index, f"{prefix}_phychem.csv")
    print(f"✅  Finished feature extraction for {prefix}")

# --------------------------------------------------------------------- #
def main():
    x_train = pd.read_csv("/home/andy/andy/Inflam_NP/preprocessing/Coconut_NP_similar_modified_x_train.csv", index_col=0)
    x_test  = pd.read_csv("/home/andy/andy/Inflam_NP/preprocessing/Coconut_NP_similar_modified_x_test.csv", index_col=0)

    process_and_save_features(x_train, "canonical_smiles", "Coconut_NP_similar_modified_x_train")
    process_and_save_features(x_test,  "canonical_smiles", "Coconut_NP_similar_modified_x_test")

if __name__ == "__main__":
    main()
