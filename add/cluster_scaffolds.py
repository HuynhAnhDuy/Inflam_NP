import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
import numpy as np

# =========================
# 1) INPUT CONFIG
# =========================
INPUT_CSV = "scaffold_shap_summary_test_NP.csv"
ID_COL = "ID"
SCAFF_COL = "scaffold"

OUT_ONE = "cluster_representatives_with_id_code2.csv"

# Clustering
TANIMOTO_THRESHOLD = 0.7
RADIUS = 2        # ECFP4
NBITS = 2048

# Optional: if you have a SHAP column per row/scaffold, put its name here.
# If None => tie-break uses heavy atoms only.
SHAP_COL = "mean_shap"   # e.g., "mean_shap" or "shap_value"

# Tie handling:
# - Consider two candidates "tied" in medoid score if difference <= EPS
EPS = 1e-6


# =========================
# 2) UTILS
# =========================
def safe_str(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    return s if s != "" else None


def mol_from_smiles(smi: str):
    try:
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None


def heavy_atoms(smi: str) -> int:
    m = mol_from_smiles(smi)
    return int(m.GetNumHeavyAtoms()) if m is not None else -1


def pick_id_for_scaffold(df_in: pd.DataFrame, scaffold: str) -> str:
    """
    Pick representative ID for the representative scaffold from original file.
    If SHAP_COL provided: pick ID with max SHAP among rows having this scaffold.
    Else: pick first ID occurrence.
    """
    sub = df_in[df_in[SCAFF_COL] == scaffold].copy()
    if len(sub) == 0:
        return None

    if SHAP_COL is not None and SHAP_COL in sub.columns:
        sub = sub.dropna(subset=[SHAP_COL])
        if len(sub) > 0:
            sub = sub.sort_values(SHAP_COL, ascending=False)
            return str(sub.iloc[0][ID_COL])

    # fallback: first occurrence
    return str(sub.iloc[0][ID_COL])


def get_scaffold_shap(df_in: pd.DataFrame, scaffold: str) -> float:
    """
    A single SHAP score for a scaffold (used for tie-break).
    If multiple rows per scaffold exist, default to max SHAP.
    You can change max -> mean if you prefer.
    """
    if SHAP_COL is None or SHAP_COL not in df_in.columns:
        return np.nan
    sub = df_in[df_in[SCAFF_COL] == scaffold][SHAP_COL].dropna()
    if len(sub) == 0:
        return np.nan
    return float(sub.max())


def select_medoid_with_tiebreak(cluster_scaffolds, cluster_fps, df_in):
    """
    Select representative scaffold using:
      1) Medoid: max average Tanimoto similarity within cluster
      2) Tie-break: higher scaffold SHAP (if available)
      3) Tie-break: larger heavy atom count
    """
    n = len(cluster_scaffolds)
    if n == 1:
        return cluster_scaffolds[0], 1.0  # avg sim = 1 by definition (single item)

    avg_sims = []
    for i in range(n):
        # similarity to all members (including itself) then average
        sims = DataStructs.BulkTanimotoSimilarity(cluster_fps[i], cluster_fps)
        avg_sims.append(float(np.mean(sims)))

    max_avg = max(avg_sims)
    # candidates within EPS of best medoid score
    cand_idx = [i for i, s in enumerate(avg_sims) if (max_avg - s) <= EPS]

    if len(cand_idx) == 1:
        i = cand_idx[0]
        return cluster_scaffolds[i], avg_sims[i]

    # ---- Tie-break #1: scaffold-level SHAP (higher is better) ----
    if SHAP_COL is not None and SHAP_COL in df_in.columns:
        cand = []
        for i in cand_idx:
            scf = cluster_scaffolds[i]
            scf_shap = get_scaffold_shap(df_in, scf)
            cand.append((scf_shap, i))
        # if all NaN, skip to heavy atoms
        if not all(np.isnan(x[0]) for x in cand):
            # sort by shap desc, keep only top shap (handle NaN as -inf)
            cand_sorted = sorted(cand, key=lambda t: (-np.nan_to_num(t[0], nan=-1e18), t[1]))
            best_shap = cand_sorted[0][0]
            top = [i for (sh, i) in cand_sorted if (np.isnan(best_shap) and np.isnan(sh)) or (not np.isnan(best_shap) and abs(sh - best_shap) <= EPS)]
            if len(top) == 1:
                i = top[0]
                return cluster_scaffolds[i], avg_sims[i]
            cand_idx = top  # still tied -> go next tie-break

    # ---- Tie-break #2: heavy atoms (larger is better) ----
    ha = [(heavy_atoms(cluster_scaffolds[i]), i) for i in cand_idx]
    ha_sorted = sorted(ha, key=lambda t: (-t[0], t[1]))
    best_i = ha_sorted[0][1]
    return cluster_scaffolds[best_i], avg_sims[best_i]


# =========================
# 3) LOAD + CLEAN
# =========================
df = pd.read_csv(INPUT_CSV)

if ID_COL not in df.columns:
    raise ValueError(f"Missing ID column '{ID_COL}'. Found: {list(df.columns)}")
if SCAFF_COL not in df.columns:
    raise ValueError(f"Missing scaffold column '{SCAFF_COL}'. Found: {list(df.columns)}")
if SHAP_COL is not None and SHAP_COL not in df.columns:
    raise ValueError(f"Missing SHAP column '{SHAP_COL}'. Found: {list(df.columns)}")

df[ID_COL] = df[ID_COL].apply(safe_str)
df[SCAFF_COL] = df[SCAFF_COL].apply(safe_str)

df_in = df.dropna(subset=[ID_COL, SCAFF_COL]).copy()

print("Total input rows:", len(df))
print("Valid rows (have ID + scaffold):", len(df_in))

# =========================
# 4) UNIQUE SCAFFOLDS -> MOL + FP
# =========================
unique_scaffolds = df_in[SCAFF_COL].drop_duplicates().tolist()

valid_scaffolds, mols = [], []
for smi in unique_scaffolds:
    m = mol_from_smiles(smi)
    if m is not None:
        valid_scaffolds.append(smi)
        mols.append(m)

print("Unique scaffolds:", len(unique_scaffolds))
print("Valid scaffolds (RDKit parsed):", len(valid_scaffolds))
print("Invalid scaffold SMILES:", len(unique_scaffolds) - len(valid_scaffolds))

fps = [AllChem.GetMorganFingerprintAsBitVect(m, RADIUS, nBits=NBITS) for m in mols]

# =========================
# 5) BUTINA CLUSTERING
# =========================
dists = []
nfps = len(fps)
for i in range(1, nfps):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
    dists.extend([1 - x for x in sims])

clusters = Butina.ClusterData(dists, nfps, TANIMOTO_THRESHOLD, isDistData=True)
print("Number of clusters:", len(clusters))

# =========================
# 6) ONE OUTPUT: medoid + tie-break (SHAP -> heavy atoms) + representative ID
# =========================
rows = []
for cid, cluster in enumerate(clusters):
    idxs = list(cluster)
    cluster_scaffolds = [valid_scaffolds[i] for i in idxs]
    cluster_fps = [fps[i] for i in idxs]

    rep_scaffold, rep_avg_sim = select_medoid_with_tiebreak(cluster_scaffolds, cluster_fps, df_in)
    rep_id = pick_id_for_scaffold(df_in, rep_scaffold)

    rows.append({
        "cluster_id": cid,
        "n_scaffolds_in_cluster": len(cluster_scaffolds),
        "representative_scaffold": rep_scaffold,
        "representative_ID": rep_id,
        "medoid_avg_tanimoto": rep_avg_sim,
        "representative_scaffold_shap": (get_scaffold_shap(df_in, rep_scaffold) if SHAP_COL is not None else np.nan),
        "representative_scaffold_heavy_atoms": heavy_atoms(rep_scaffold),
    })

out = pd.DataFrame(rows).sort_values(["n_scaffolds_in_cluster", "medoid_avg_tanimoto"], ascending=False)
out.to_csv(OUT_ONE, index=False, encoding="utf-8-sig")
print("Saved:", OUT_ONE)

print("\nPreview:")
print(out.head(15).to_string(index=False))
