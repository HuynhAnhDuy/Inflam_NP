# -*- coding: utf-8 -*-
import os, gzip, time
import pandas as pd
import numpy as np
from typing import Optional, List
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.inchi import MolToInchiKey
from rdkit import RDLogger
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')  # giảm log RDKit

# ================== CONFIG ==================
POS_SCAFFOLDS_CSV = "/home/andy/andy/Inflam_NP/Scaffold_identify/shap_scaffold_analysis_XGB_20250915_143444/scaffold_shap_summary.csv"  # cột: scaffold
POS_SCAFF_COL     = "scaffold"

NP_INPUT_PATHS = [
    "coconut_csv-09-2025_clean.csv",
    # "/data/NPASS/npass_structures.csv",
    # "/data/COCONUT/Coconut-gz.sdf.gz",
]

# CSV (COCONUT) dùng 'canonical_smiles'
NP_ID_COL       = "compound_id"         # nếu CSV không có, script sẽ tự tạo
NP_SMILES_COLS  = ["canonical_smiles"]  # ưu tiên 'canonical_smiles'

# Similarity (trên Murcko-scaffold fingerprints)
N_BITS        = 2048
RADIUS        = 2
SIM_THRESHOLD = 0.80   # 0.7–0.85 thường hợp lý

# Output
OUT_EXACT_ONLY_CSV   = "NP_candidates_exact_only.csv"
OUT_SIMILAR_ONLY_CSV = "NP_candidates_similar_only.csv"

# Verbose
PRINT_EVERY = 5000  # in log mỗi N dòng khi xử lý lớn
# ============================================


# ----------------- Helpers ------------------
def log(msg: str):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")

def mol_from_smiles(smi: str) -> Optional[Chem.Mol]:
    if not isinstance(smi, str) or not smi:
        return None
    return Chem.MolFromSmiles(smi)

def standardize_mol(m: Optional[Chem.Mol]) -> Optional[Chem.Mol]:
    if m is None: return None
    try:
        params = rdMolStandardize.CleanupParameters()
        m = rdMolStandardize.Cleanup(m, params)
        m = rdMolStandardize.LargestFragmentChooser().choose(m)
        m = rdMolStandardize.Uncharger().uncharge(m)
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None

def murcko_smiles_from_mol(m: Optional[Chem.Mol]) -> Optional[str]:
    if m is None: return None
    try:
        scf = MurckoScaffold.GetScaffoldForMol(m)
        if scf is None: return None
        return Chem.MolToSmiles(scf, isomericSmiles=False, canonical=True)
    except Exception:
        return None

def fp_from_smiles(smi: str, radius=RADIUS, nbits=N_BITS):
    m = mol_from_smiles(smi)
    if m is None: return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nbits)
    except Exception:
        return None

def tanimoto(fp1, fp2, default=0.0):
    if fp1 is None or fp2 is None: return default
    return TanimotoSimilarity(fp1, fp2)

def read_sdf_or_gz(path: str) -> pd.DataFrame:
    """Đọc .sdf hoặc .sdf.gz -> DataFrame[cid, canonical_SMILES] (đã chuẩn hoá)"""
    start = time.time()
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            data = f.read()
        tmp = "_tmp_np_struct.sdf"
        with open(tmp, "wb") as w:
            w.write(data)
        supplier = Chem.SDMolSupplier(tmp, sanitize=False, removeHs=True)
        try: os.remove(tmp)
        except: pass
    else:
        supplier = Chem.SDMolSupplier(path, sanitize=False, removeHs=True)

    rows = []
    cnt = 0
    log(f"Parsing SDF: {path}")
    for mol in tqdm(supplier, desc="SDF->canonical_SMILES", unit="mol"):
        if mol is None: 
            continue
        try:
            m = Chem.Mol(mol)
            m = standardize_mol(m)
            if m is None: 
                continue
            smi = Chem.MolToSmiles(m, isomericSmiles=False, canonical=True)
            cid = mol.GetProp("_Name") if mol.HasProp("_Name") else None
            rows.append({ "compound_id": cid, "canonical_SMILES": smi })
            cnt += 1
        except Exception:
            continue
    log(f"SDF parsed: {cnt} rows in {time.time()-start:.1f}s")
    return pd.DataFrame(rows)

def _pick_smiles_col(df: pd.DataFrame) -> str:
    for c in NP_SMILES_COLS:
        if c in df.columns:
            return c
    raise ValueError(f"CSV thiếu cột SMILES. Thử một trong: {NP_SMILES_COLS}")

def read_np_path(path: str) -> pd.DataFrame:
    """Đọc 1 nguồn NP (CSV hoặc SDF), chuẩn hoá canonical_SMILES và gán InChIKey để khử trùng"""
    ext = os.path.splitext(path)[1].lower()
    if ext in [".sdf", ".gz"]:
        df = read_sdf_or_gz(path)
    else:
        log(f"Reading CSV: {path}")
        df_raw = pd.read_csv(path)
        smiles_col = _pick_smiles_col(df_raw)  # ưu tiên 'canonical_smiles'
        if NP_ID_COL not in df_raw.columns:
            df_raw[NP_ID_COL] = np.arange(1, len(df_raw)+1)

        cleaned = []
        skipped = 0
        it = df_raw[[NP_ID_COL, smiles_col]].itertuples(index=False, name=None)
        log(f"Standardizing molecules (CSV) ...")
        for idx, (cid, smi) in enumerate(tqdm(it, total=len(df_raw), unit="row")):
            m = standardize_mol(mol_from_smiles(smi))
            if m is None:
                skipped += 1
                continue
            csmi = Chem.MolToSmiles(m, isomericSmiles=False, canonical=True)
            cleaned.append({ "compound_id": cid, "canonical_SMILES": csmi })
            if (idx+1) % PRINT_EVERY == 0:
                log(f"  processed {idx+1}/{len(df_raw)} (skipped={skipped})")
        df = pd.DataFrame(cleaned)
        log(f"[{os.path.basename(path)}] cleaned={len(df)} skipped={skipped}")

    # khử trùng theo InChIKey
    log("Deduplicating by InChIKey ...")
    uniq = []
    seen = set()
    for r in tqdm(df.itertuples(index=False), total=len(df), unit="mol"):
        m = mol_from_smiles(getattr(r, "canonical_SMILES"))
        if m is None: 
            continue
        try:
            ik = MolToInchiKey(m)
        except Exception:
            ik = None
        if ik and ik not in seen:
            seen.add(ik)
            uniq.append({ 
                "compound_id": getattr(r, "compound_id"),
                "canonical_SMILES": getattr(r, "canonical_SMILES"),
                "InChIKey": ik 
            })
    log(f"Unique molecules: {len(uniq)}")
    return pd.DataFrame(uniq)

# --------------- Load data ------------------
t0 = time.time()
log("Loading positive scaffolds ...")
pos_df = pd.read_csv(POS_SCAFFOLDS_CSV)
if POS_SCAFF_COL not in pos_df.columns:
    raise ValueError(f"File {POS_SCAFFOLDS_CSV} thiếu cột '{POS_SCAFF_COL}'")
positive_scaffolds = pos_df[POS_SCAFF_COL].dropna().astype(str).unique().tolist()
log(f"Positive scaffolds: {len(positive_scaffolds)}")

log("Precomputing fingerprints for positive scaffolds ...")
pos_fps = []
for scf in tqdm(positive_scaffolds, desc="Pos scaff FP", unit="scf"):
    pos_fps.append(fp_from_smiles(scf))

# NP sources
frames: List[pd.DataFrame] = []
for p in NP_INPUT_PATHS:
    log(f"Reading NP source: {p}")
    try:
        frames.append(read_np_path(p))
    except Exception as e:
        log(f"WARN: skip {p} ({e})")

if len(frames) == 0:
    raise RuntimeError("Không có dữ liệu NP hợp lệ. Kiểm tra NP_INPUT_PATHS.")

NP = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["InChIKey"]).reset_index(drop=True)
log(f"Loaded NP molecules (dedup by InChIKey): {len(NP)}")

# --------------- Scaffold & Flags -----------
log("Computing Murcko scaffolds for NP ...")
def _row_to_scaffold(csmi: str) -> Optional[str]:
    return murcko_smiles_from_mol(mol_from_smiles(csmi))

NP["scaffold"] = [ _row_to_scaffold(s) for s in tqdm(NP["canonical_SMILES"], total=len(NP), unit="mol") ]

log("Flagging EXACT scaffold hits ...")
pos_set = set(positive_scaffolds)
NP["exact_scaffold_hit"] = NP["scaffold"].isin(pos_set)

log("Flagging SIMILAR scaffold hits ...")
def similar_to_positive(scf_smi: Optional[str], thr=SIM_THRESHOLD) -> bool:
    if not scf_smi: 
        return False
    fp = fp_from_smiles(scf_smi)
    if fp is None: 
        return False
    for pfp in pos_fps:
        if pfp is None: 
            continue
        if tanimoto(fp, pfp) >= thr:
            return True
    return False

NP["similar_scaffold_hit"] = [
    similar_to_positive(s, SIM_THRESHOLD) for s in tqdm(NP["scaffold"], total=len(NP), unit="scaf")
]

# --------------- Export ---------------------
cols = [c for c in ["compound_id","InChIKey","canonical_SMILES","scaffold",
                    "exact_scaffold_hit","similar_scaffold_hit"] if c in NP.columns]

log("Preparing EXACT file ...")
hits_exact = NP[NP["exact_scaffold_hit"]].copy()
hits_exact["Note"] = "Exact"
hits_exact = hits_exact[cols + ["Note"]]
hits_exact.to_csv(OUT_EXACT_ONLY_CSV, index=False)
log(f"Saved {OUT_EXACT_ONLY_CSV} rows={len(hits_exact)}")

log("Preparing SIMILAR file ...")
hits_similar = NP[NP["similar_scaffold_hit"] & ~NP["exact_scaffold_hit"]].copy()
hits_similar["Note"] = "Similar"
hits_similar = hits_similar[cols + ["Note"]]
hits_similar.to_csv(OUT_SIMILAR_ONLY_CSV, index=False)
log(f"Saved {OUT_SIMILAR_ONLY_CSV} rows={len(hits_similar)}")

log(f"All done in {time.time()-t0:.1f}s")
log(f"Params: SIM_THRESHOLD={SIM_THRESHOLD}, N_BITS={N_BITS}, RADIUS={RADIUS}")
