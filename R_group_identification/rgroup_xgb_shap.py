# pip install rdkit-pypi xgboost shap scikit-learn pandas numpy

import pandas as pd, numpy as np, shap, warnings
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdmolfiles import CanonicalRankAtoms
from collections import deque

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# === LOAD DATASET ===
df = pd.read_csv("/home/andy/andy/Inflam_NP/R_group_identification/InFlam_full.csv")
df = df.dropna(subset=["canonical_smiles", "Label"])
df["Label"] = df["Label"].astype(int)

# === GET SCAFFOLD ===
def get_scaffold(mol):
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf and scaf.GetNumAtoms():
            Chem.SanitizeMol(scaf)
            return scaf
    except:
        pass

# === INTERPRET R-GROUPS ===
def interpret_rgroup(smi):
    """Return a human-readable note for common R-groups"""
    if smi in ["C[*]", "*C"]:
        return "Methyl (-CH3)"
    elif smi in ["O[*]", "*O"]:
        return "Hydroxyl (-OH)"
    elif smi in ["N[*]", "*N"]:
        return "Amino (-NH2)"
    elif smi in ["Cl[*]", "*Cl"]:
        return "Chloro (-Cl)"
    elif smi in ["Br[*]", "*Br"]:
        return "Bromo (-Br)"
    elif smi in ["F[*]", "*F"]:
        return "Fluoro (-F)"
    elif "C(=O)O" in smi:
        return "Carboxyl (-COOH)"
    elif "N(=O)(=O)" in smi:
        return "Nitro (-NO2)"
    elif "C(F)(F)F" in smi:
        return "Trifluoromethyl (-CF3)"
    elif "C#N" in smi:
        return "Cyano (-CN)"
    elif "S(=O)(=O)N" in smi:
        return "Sulfonamide (-SO2NH2)"
    else:
        return "Other/complex group"


# === INTERPRET SCAFFOLD CORE ===
def interpret_scaffold(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return "Unknown scaffold"
        if Chem.MolFromSmarts("c1ccccc1").HasSubstructMatch(m):
            return "Benzene core"
        if Chem.MolFromSmarts("c1ccncc1").HasSubstructMatch(m):
            return "Pyridine core"
        if Chem.MolFromSmarts("c1ccc2[nH]ccc2c1").HasSubstructMatch(m):
            return "Indole core"
        if Chem.MolFromSmarts("c1ccc2ncccc2c1").HasSubstructMatch(m):
            return "Quinoline core"
        if Chem.MolFromSmarts("C1CCCCC1").HasSubstructMatch(m):
            return "Cyclohexane core"
        return "Other/complex scaffold"
    except:
        return "Invalid scaffold"

# === EXTRACT R-GROUPS ===
def extract_r_groups(mol, scaf):
    match = mol.GetSubstructMatch(scaf)
    if not match: 
        return []
    scaf_ranks = CanonicalRankAtoms(scaf)
    parent_atoms = set(match)
    r_groups = []

    for scaf_idx, p_idx in dict(enumerate(match)).items():
        rpos = f"R@{scaf_ranks[scaf_idx]}"
        atom = mol.GetAtomWithIdx(p_idx)
        for nb in atom.GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx in parent_atoms: 
                continue
            visited, q = {nb_idx}, deque([nb_idx])
            while q:
                cur = q.popleft()
                for nb2 in mol.GetAtomWithIdx(cur).GetNeighbors():
                    i = nb2.GetIdx()
                    if i not in visited and i not in parent_atoms:
                        visited.add(i)
                        q.append(i)

            rw, map_idx = Chem.RWMol(), {}
            for i in sorted(visited):
                a = mol.GetAtomWithIdx(i)
                na = Chem.Atom(a.GetAtomicNum())
                na.SetFormalCharge(a.GetFormalCharge())
                map_idx[i] = rw.AddAtom(na)

            for i in visited:
                for b in mol.GetAtomWithIdx(i).GetBonds():
                    i1, i2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                    if i1 in visited and i2 in visited and i1 != i2:
                        ni, nj = map_idx[i1], map_idx[i2]
                        if rw.GetBondBetweenAtoms(ni, nj) is None:
                            rw.AddBond(ni, nj, b.GetBondType())

            dummy = Chem.Atom(0)
            dummy_idx = rw.AddAtom(dummy)
            if rw.GetNumAtoms() > 1:
                rw.AddBond(dummy_idx, 0, Chem.BondType.SINGLE)

            r_frag = rw.GetMol()
            smiles = Chem.MolToSmiles(r_frag, canonical=True)

            r_groups.append({
                "R_pos": rpos,
                "R_group_smiles": smiles,
                "R_group_note": interpret_rgroup(smiles)
            })
    return r_groups

# === STEP 1: Extract scaffolds ===
mol_scaffolds = []
for i, row in df.iterrows():
    mol = Chem.MolFromSmiles(row["canonical_smiles"])
    if mol is None:
        continue
    scaf = get_scaffold(mol)
    core_id = Chem.MolToSmiles(scaf, True) if scaf else "NO_SCAF"
    mol_scaffolds.append({
        "mol_idx": i, "mol": mol, "core_id": core_id, "Label": row["Label"]
    })

df_scaf = pd.DataFrame(mol_scaffolds)

# === STEP 2: Keep scaffolds with >= 10 compounds ===
scaf_counts = df_scaf.groupby("core_id")["mol_idx"].nunique().reset_index(name="n_compounds")
valid_scafs = scaf_counts.query("n_compounds >= 10")["core_id"].tolist()

df_scaf = df_scaf[df_scaf["core_id"].isin(valid_scafs)].reset_index(drop=True)

print(f"Valid scaffolds (>=10 compounds): {len(valid_scafs)}")
print("Remaining molecules:", len(df_scaf))

# === STEP 3: Extract R-groups for valid scaffolds ===
records = []
for _, row in df_scaf.iterrows():
    mol, core_id, label, midx = row["mol"], row["core_id"], row["Label"], row["mol_idx"]

    if core_id == "NO_SCAF":
        records.append({
            "mol_idx": midx, "core_id": core_id, "R_pos": "R@NA",
            "R_group_smiles": Chem.MolToSmiles(mol),
            "R_group_note": "Whole molecule",
            "Label": label
        })
        continue

    scaf = Chem.MolFromSmiles(core_id)
    rgs = extract_r_groups(mol, scaf)

    if not rgs:
        records.append({
            "mol_idx": midx, "core_id": core_id, "R_pos": "R@NONE",
            "R_group_smiles": "[H]",
            "R_group_note": "No substituent",
            "Label": label
        })
    else:
        for r in rgs:
            records.append({"mol_idx": midx, "core_id": core_id, **r, "Label": label})

df_rg = pd.DataFrame(records)

# Add scaffold plain-text interpretation
df_rg["core_note"] = df_rg["core_id"].apply(interpret_scaffold)

# === DESCRIPTORS FOR R-GROUPS ===
def r_desc(smi):
    try:
        m = Chem.MolFromSmiles(smi.replace("[*]",""))
        return [
            Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m),
            Descriptors.NumHDonors(m), Descriptors.NumHAcceptors(m),
            rdMolDescriptors.CalcNumAliphaticRings(m),
            rdMolDescriptors.CalcNumAromaticRings(m),
            rdMolDescriptors.CalcFractionCSP3(m)
        ] if m else [np.nan]*8
    except:
        return [np.nan]*8

desc_cols = ["MW", "cLogP", "TPSA", "HBD", "HBA", "AliRings", "AroRings", "fSP3"]
df_rg[desc_cols] = df_rg["R_group_smiles"].apply(lambda x: pd.Series(r_desc(x)))
df_rg = df_rg.dropna(subset=desc_cols)

# === TRAINING + SHAP ===
X = df_rg[desc_cols + ["R_pos", "core_id"]]
y = df_rg["Label"].values
pre = ColumnTransformer([
    ("num", "passthrough", desc_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["R_pos", "core_id"])
])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_shap, oof_pred = np.zeros(len(df_rg)), np.zeros(len(df_rg))

for fold, (tr, te) in enumerate(skf.split(X, y)):
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
    clf = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, 
                        subsample=0.9, colsample_bytree=0.8, n_jobs=4,
                        reg_lambda=1.0, eval_metric="logloss", random_state=fold)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(Xtr, ytr)
    oof_pred[te] = pipe.predict_proba(Xte)[:,1]

    booster = pipe.named_steps["clf"]
    Xte_t = pipe.named_steps["pre"].transform(Xte)
    explainer = shap.TreeExplainer(booster)
    shap_vals = np.zeros_like(Xte_t)
    for start in range(0, len(Xte_t), 500):
        end = min(start+500, len(Xte_t))
        shap_vals[start:end] = explainer.shap_values(Xte_t[start:end])
    oof_shap[te] = shap_vals.sum(axis=1)

# === RESULTS & SUMMARIES ===
df_rg["shap_sum"], df_rg["pred_proba"] = oof_shap, oof_pred
print("ROC-AUC:", roc_auc_score(y, oof_pred))
print("PR-AUC :", average_precision_score(y, oof_pred))

# 1. Detailed per scaffold/position/group
summary = (
    df_rg.groupby(["core_id", "core_note", "R_pos", "R_group_smiles", "R_group_note"])
         .agg(mean_shap=("shap_sum", "mean"),
              n=("Label", "size"),
              frac_pos=("Label", "mean"))
         .reset_index()
         .sort_values("mean_shap", ascending=False)
)
summary.to_csv("XGB_Scaffold_Rgroup_summary.csv", index=False, encoding="utf-8-sig")

# 2. Top positive-impact R-groups per scaffold
summary.query("mean_shap > 0 and n >= 10").head(50).to_csv(
    "XGB_Scaffold_Rgroup_top_positive.csv", index=False, encoding="utf-8-sig"
)

# 3. Overall R-group impact (across scaffolds)
overall = (
    df_rg.groupby(["R_group_smiles", "R_group_note"])
         .agg(mean_shap=("shap_sum", "mean"),
              n=("Label", "size"),
              frac_pos=("Label", "mean"))
         .reset_index()
         .sort_values("mean_shap", ascending=False)
)
overall.to_csv("XGB_Rgroup_overall_summary.csv", index=False, encoding="utf-8-sig")

# 4. Top positive-impact R-groups overall
overall.query("mean_shap > 0 and n >= 20").head(50).to_csv(
    "XGB_Rgroup_overall_top_positive.csv", index=False, encoding="utf-8-sig"
)

print("Done.")
