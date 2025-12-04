import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.ML.Cluster import Butina

# ====================== CONFIG ======================
# Đường dẫn file input/output
INPUT_CSV  = "/home/andy/andy/Inflam_NP/Scaffold_clustering/scaffold_shap_summary_test.csv"   # file 910 scaffold dương
OUTPUT_CLUSTERED = "scaffolds_clustered.csv"      # file có thêm cluster_id
OUTPUT_TOP20     = "scaffolds_top20_diverse.csv"  # top 20 scaffold để minh hoạ

# Tên cột trong file CSV của anh
SMILES_COL = "scaffold_smiles"   # cột chứa scaffold (Bemis–Murcko) SMILES
SHAP_COL   = "mean_shap"         # cột chứa SHAP value (vd: mean SHAP per scaffold)

# Tham số fingerprint & clustering
FP_NBITS   = 2048       # số bit cho ECFP4
FP_RADIUS  = 4          # ECFP4 = radius 2
BUTINA_CUTOFF = 0.4    # cutoff cho clustering (0.3–0.4 thường hợp lý)
TOP_N      = 20         # số scaffold muốn chọn minh hoạ
# ====================================================


# 1. Đọc data
df = pd.read_csv(INPUT_CSV)

# Kiểm tra cột
if SMILES_COL not in df.columns:
    raise ValueError(f"Không tìm thấy cột '{SMILES_COL}' trong file CSV.")
if SHAP_COL not in df.columns:
    raise ValueError(f"Không tìm thấy cột '{SHAP_COL}' trong file CSV.")

# Giữ lại các hàng có SMILES & SHAP hợp lệ
df = df.copy()
df = df[df[SMILES_COL].notna() & df[SHAP_COL].notna()].reset_index(drop=True)

print(f"Số dòng sau khi lọc NA: {len(df)}")

# 2. Chuyển scaffold SMILES → RDKit Mol
mols = []
valid_idx = []
for i, smi in enumerate(df[SMILES_COL].astype(str)):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        mols.append(mol)
        valid_idx.append(i)
    else:
        print(f"[Warning] Không parse được SMILES tại dòng {i}: {smi}")

if len(mols) == 0:
    raise ValueError("Không có Mol hợp lệ sau khi parse SMILES.")

# Tạo một dataframe chỉ chứa các dòng Mol hợp lệ
df_valid = df.loc[valid_idx].reset_index(drop=True)
print(f"Số scaffold hợp lệ: {len(df_valid)}")

# 3. Tạo ECFP4 fingerprints cho từng scaffold
fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=FP_RADIUS, nBits=FP_NBITS)
       for m in mols]

# 4. Tính ma trận khoảng cách (1 - Tanimoto similarity) cho Butina
#    Data format cho Butina: list distances của cặp (1,0), (2,0),(2,1), (3,0),(3,1),(3,2), ...
dists = []
nfps = len(fps)
for i in range(1, nfps):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
    dists.extend([1 - x for x in sims])

print("Đã tính xong ma trận khoảng cách cho Butina clustering.")

# 5. Thực hiện Butina clustering
clusters = Butina.ClusterData(dists, nfps, BUTINA_CUTOFF, isDistData=True)
print(f"Số cụm thu được: {len(clusters)}")

# 6. Gán cluster_id cho từng scaffold
# RDKit Butina trả về tuple index; phần tử đầu tiên là "centroid" (đại diện cluster)
cluster_id_array = np.empty(nfps, dtype=int)
cluster_centroid_idx = []

for cid, cluster in enumerate(clusters):
    for idx in cluster:
        cluster_id_array[idx] = cid
    # index đầu của cluster được coi là centroid
    centroid_idx = cluster[0]
    cluster_centroid_idx.append(centroid_idx)

# Tạo cột cluster_id và is_centroid
df_valid["cluster_id"] = cluster_id_array
df_valid["is_centroid"] = False
df_valid.loc[cluster_centroid_idx, "is_centroid"] = True

# Thêm rank SHAP toàn cục
df_valid = df_valid.sort_values(by=SHAP_COL, ascending=False).reset_index(drop=True)
df_valid["global_rank_by_SHAP"] = np.arange(1, len(df_valid) + 1)

# Thêm rank SHAP trong từng cluster
df_valid["cluster_rank_by_SHAP"] = (
    df_valid
    .sort_values(by=[ "cluster_id", SHAP_COL ], ascending=[True, False])
    .groupby("cluster_id")
    .cumcount() + 1
)

# 7. Lưu file đầy đủ kèm cluster
df_valid.to_csv(OUTPUT_CLUSTERED, index=False)
print(f"Đã lưu file phân cụm: {OUTPUT_CLUSTERED}")


# 8. Chọn Top 20 scaffold "đa dạng cluster" (mỗi cluster tối đa 1 scaffold)
# Ý tưởng:
#  - Sắp df_valid theo SHAP giảm dần
#  - Duyệt từ trên xuống, nếu cluster đó chưa được chọn → lấy scaffold vào danh sách
#  - Dừng khi đủ TOP_N hoặc hết cluster

df_sorted = df_valid.sort_values(by=SHAP_COL, ascending=False).reset_index(drop=True)

selected_indices = []
used_clusters = set()

for idx, row in df_sorted.iterrows():
    cid = row["cluster_id"]
    if cid not in used_clusters:
        selected_indices.append(idx)
        used_clusters.add(cid)
        if len(selected_indices) >= TOP_N:
            break

# Nếu số cluster < TOP_N → sẽ thiếu, thì lấy thêm scaffold SHAP cao nhất
# mà chưa được chọn, bất kể cluster (cho đủ TOP_N)
if len(selected_indices) < TOP_N:
    print(f"Chỉ có {len(selected_indices)} cluster khác nhau. "
          f"Sẽ lấy thêm scaffold để đủ {TOP_N} (có thể trùng cluster).")
    already_selected = set(selected_indices)
    for idx, row in df_sorted.iterrows():
        if idx not in already_selected:
            selected_indices.append(idx)
            already_selected.add(idx)
            if len(selected_indices) >= TOP_N:
                break

# Tạo dataframe top N
df_topN = df_sorted.loc[selected_indices].copy()
df_topN = df_topN.sort_values(by=SHAP_COL, ascending=False).reset_index(drop=True)

# Gán rank riêng cho top N
df_topN["topN_rank_by_SHAP"] = np.arange(1, len(df_topN) + 1)

# 9. Lưu top 20
df_topN.to_csv(OUTPUT_TOP20, index=False)
print(f"Đã lưu Top {TOP_N} scaffold: {OUTPUT_TOP20}")

# 10. Một chút summary
print("\nTóm tắt số lượng scaffold / cluster (top 10 cluster đầu):")
cluster_counts = df_valid["cluster_id"].value_counts().sort_index()
print(cluster_counts.head(10))

print("\nTop scaffold theo SHAP trong từng cluster (5 cluster đầu):")
for cid in sorted(df_valid["cluster_id"].unique())[:5]:
    sub = df_valid[df_valid["cluster_id"] == cid].sort_values(by=SHAP_COL, ascending=False)
    top_row = sub.iloc[0]
    print(f"- Cluster {cid}: size={len(sub)}, best_SHAP={top_row[SHAP_COL]:.4f}, "
          f"SMILES={top_row[SMILES_COL]}")
