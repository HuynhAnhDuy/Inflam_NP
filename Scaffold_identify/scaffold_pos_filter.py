import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity

# ======= Tham số =======
SHAP_CSV       = "/home/andy/andy/Inflam_NP/Scaffold_identify/shap_scaffold_split_XGB_20250915_151119/scaffold_positive_overlap.csv"  # cột: scaffold, mean_shap
DATA_CSV       = "3.InFlamNat_SHAP_with_scaffolds.csv"   # có: canonical_smiles, scaffold (hoặc scaffold_generic)
SCAFF_COL      = "scaffold"          # đổi thành "scaffold_generic" nếu SHAP dùng generic
MIN_COVERAGE   = 5
TOPN           = 10
SIM_THRESHOLD  = 0.8
N_BITS         = 2048
RADIUS         = 2

# ======= ECFP cho Murcko-SMILES =======
def ecfp_murcko_bvect(scf_smi, radius=RADIUS, nbits=N_BITS):
    if not isinstance(scf_smi, str) or not scf_smi:
        return None
    m = Chem.MolFromSmiles(scf_smi)
    if m is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nbits)
    except Exception:
        return None

def is_similar_to_selected(cand_bv, selected_bvects, thr=SIM_THRESHOLD):
    if cand_bv is None:
        return False
    for bv in selected_bvects:
        if bv is not None and TanimotoSimilarity(cand_bv, bv) >= thr:
            return True
    return False

# ======= 1) Load =======
df_shap = pd.read_csv(SHAP_CSV)       # cần cột: scaffold | mean_shap
df_all  = pd.read_csv(DATA_CSV)       # cần cột: canonical_smiles | SCAFF_COL (scaffold hoặc scaffold_generic)

# Gom trùng nếu shap file có lặp scaffold
df_shap = (df_shap.groupby("scaffold", as_index=False)["mean_shap"]
                 .mean())

# ======= 2) Coverage theo SCAFF_COL =======
cov = (df_all.groupby(SCAFF_COL)["canonical_smiles"]
             .count().reset_index()
             .rename(columns={"canonical_smiles": "coverage"}))

# ======= 3) Merge & lọc dương =======
df = df_shap.merge(cov, left_on="scaffold", right_on=SCAFF_COL, how="left")
df["coverage"] = df["coverage"].fillna(0).astype(int)

# Giữ scaffold có mean_shap > 0 và đủ coverage
df = df[(df["mean_shap"] > 0) & (df["coverage"] >= MIN_COVERAGE)].copy()

# ======= 4) Priority (không cần p-values) =======
# Bạn có thể thử các biến thể khác: mean_shap * sqrt(coverage), hoặc mean_shap ** 2 * log1p(coverage)
df["priority"] = df["mean_shap"] * np.log1p(df["coverage"])

# Sắp xếp cố định để tính diversity_flag toàn cục
df = df.sort_values(["priority", "mean_shap", "coverage"], ascending=[False, False, False]).reset_index(drop=True)

# ======= 5) Tính diversity_flag TOÀN CỤC (CỐ ĐỊNH) =======
# Tạo fingerprint một lần
df["fp"] = df["scaffold"].apply(lambda s: ecfp_murcko_bvect(s, radius=RADIUS, nbits=N_BITS))

selected_bvects = []
div_flags = []
for i, row in df.iterrows():
    cand_bv = row["fp"]
    # diverse nếu KHÔNG tương tự bất kỳ fp đã được chọn True trước đó
    diverse = not is_similar_to_selected(cand_bv, selected_bvects, thr=SIM_THRESHOLD)
    div_flags.append(diverse)
    if diverse:
        selected_bvects.append(cand_bv)

df["diversity_flag"] = div_flags

# ======= 6) Xuất file =======
# File FULL: giữ nguyên toàn bộ danh sách với flag đã cố định
out_full = "xgb_prioritized_scaffolds_with_diversity_flag_fulldata.csv"
df.drop(columns=["fp"]).to_csv(out_full, index=False)

# File TopN: chỉ lấy N dòng đầu TIÊN TỪ BẢNG ĐÃ SẮP XẾP (flag giữ nguyên, không tính lại)
out_topn = "xgb_prioritized_scaffolds_diverse_topN_fulldata.csv"
df_topn = df.head(TOPN).drop(columns=["fp"])
df_topn.to_csv(out_topn, index=False)

print("✅ Saved:")
print(" -", out_full)
print(" -", out_topn)
print("📌 Params:", {"TOPN": TOPN, "SIM_THRESHOLD": SIM_THRESHOLD,
                   "MIN_COVERAGE": MIN_COVERAGE, "SCAFF_COL": SCAFF_COL})
