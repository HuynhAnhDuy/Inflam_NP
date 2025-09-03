# pip install rdkit-pypi pandas
import pandas as pd
from rdkit import Chem

# ============== EDIT HERE ==============
PATH_COCONUT = "coconut_valid.csv"            # có canonical_smiles + InChIKey
PATH_NPASS   = "NPASSv2.0_only_valid.csv"              # có SMILES + InChIKey
PATH_MYDATA  = "inactive_samples_expanded.csv" # chỉ có SMILES
# ======================================

# ---------- helpers ----------
def pick_col(df, keywords):
    kl = [k.lower() for k in keywords]
    for c in df.columns:
        cl = c.lower()
        for k in kl:
            if k in cl:
                return c
    return None

def mol_from_smiles(s):
    if not isinstance(s, str) or not s.strip():
        return None
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    # nếu có muối/phân mảnh ".", chọn mảnh lớn nhất
    if "." in s:
        try:
            frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
        except Exception:
            frags = []
        if frags:
            m = max(frags, key=lambda x: x.GetNumAtoms())
            Chem.SanitizeMol(m)
    else:
        Chem.SanitizeMol(m)
    return m

def calc_can_smi_inchikey(smi):
    m = mol_from_smiles(smi)
    if m is None:
        return pd.Series([None, None])
    can_smi = Chem.MolToSmiles(m, canonical=True)
    ik = Chem.inchi.MolToInchiKey(m)
    return pd.Series([can_smi, ik])

# ---------- 1) Lấy InChIKey từ COCONUT ----------
coco = pd.read_csv(PATH_COCONUT, low_memory=False)
ik_c = pick_col(coco, ["InChIKey"])
if ik_c is None:
    raise ValueError("COCONUT: không thấy cột InChIKey.")
coco = coco.rename(columns={ik_c: "InChIKey"})
coco["InChIKey"] = coco["InChIKey"].astype(str).str.strip().str.upper()
np_keys_coconut = coco[["InChIKey"]].dropna().drop_duplicates()

# ---------- 2) Lấy InChIKey từ NPASS (ưu tiên tính lại từ SMILES) ----------
npass = pd.read_csv(PATH_NPASS, low_memory=False)
ik_n = pick_col(npass, ["InChIKey"])
smi_n = pick_col(npass, ["SMILES"])
if (ik_n is None) and (smi_n is None):
    raise ValueError("NPASS: cần có SMILES hoặc InChIKey.")

np_keys_npass = pd.DataFrame(columns=["InChIKey"])

# nếu có SMILES: tính InChIKey bằng RDKit
if smi_n is not None:
    tmp = npass[smi_n].apply(calc_can_smi_inchikey)
    npass["canonical_smiles_rdkit"] = tmp[0]
    npass["InChIKey_rdkit"] = tmp[1]
    np_keys_npass = pd.DataFrame({"InChIKey": npass["InChIKey_rdkit"]}).dropna().drop_duplicates()

# nếu có InChIKey sẵn: chuẩn hoá & union
if ik_n is not None:
    npass["InChIKey_provided"] = npass[ik_n].astype(str).str.strip().str.upper()
    np_keys_npass2 = pd.DataFrame({"InChIKey": npass["InChIKey_provided"]}).dropna().drop_duplicates()
    np_keys_npass = pd.concat([np_keys_npass, np_keys_npass2], ignore_index=True).drop_duplicates()

# ---------- 3) Hợp nhất NP keys (COCONUT + NPASS) ----------
np_keys = pd.concat([np_keys_coconut, np_keys_npass], ignore_index=True).dropna().drop_duplicates()
print(f"NP keys collected: {len(np_keys)}")

# ---------- 4) Xử lý file của bạn (chỉ có SMILES) ----------
mine = pd.read_csv(PATH_MYDATA, low_memory=False)
smi_m = pick_col(mine, ["SMILES"])
if smi_m is None:
    raise ValueError("MYDATA: không thấy cột SMILES.")
mine = mine.rename(columns={smi_m: "SMILES"})
tmp = mine["SMILES"].apply(calc_can_smi_inchikey)
mine["canonical_smiles_rdkit"] = tmp[0]
mine["InChIKey_rdkit"] = tmp[1]

# ---------- 5) Join để gắn cờ natural product ----------
out = mine.merge(np_keys, left_on="InChIKey_rdkit", right_on="InChIKey", how="left", indicator=True)
out["is_natural_product"] = (out["_merge"] == "both")
out = out.drop(columns=["InChIKey","_merge"])

# ---------- 6) Xuất kết quả ----------
out.to_csv("mydata_with_np_flag.csv", index=False)
out[out["is_natural_product"]].to_csv("mydata_only_np.csv", index=False)

print("Saved: mydata_with_np_flag.csv (full + cờ), mydata_only_np.csv (chỉ NP).")
print(f"NP-flagged: {out['is_natural_product'].sum()} / {len(out)}")
