import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# === 1. Đọc danh sách scaffold dương ===
df_positive = pd.read_csv("/home/andy/andy/Inflam_NP/Scaffold_identify/shap_scaffold_analysis_20250905_093742_Full/scaffold_shap_summary_fulldataset.csv")
positive_scaffolds = df_positive['scaffold'].dropna().drop_duplicates().head(5).tolist()

# === 2. Đọc dữ liệu gốc (chỉ có canonical_smiles) ===
df_data = pd.read_csv("/home/andy/andy/Inflam_NP/Scaffold_identify/3.InFlamNat_preprocess.csv")  # Chỉ cần cột 'canonical_smiles'

# === 3. Tạo scaffold từ canonical_smiles ===
def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    return None

df_data['scaffold'] = df_data['canonical_smiles'].apply(get_scaffold)

# === 4. Lọc theo 10 scaffold dương đầu tiên ===
df_filtered = df_data[df_data['scaffold'].isin(positive_scaffolds)].copy()

# === 5. Thêm scaffold_rank để đánh số thứ tự xuất hiện ===
scaffold_rank_map = {scaffold: i+1 for i, scaffold in enumerate(positive_scaffolds)}
df_filtered['scaffold_rank'] = df_filtered['scaffold'].map(scaffold_rank_map)

# === 7. Thống kê số compound mỗi scaffold ===
df_counts = df_filtered.groupby(['scaffold', 'scaffold_rank']) \
                       .size().reset_index(name='compound_count') \
                       .sort_values('scaffold_rank')

df_counts.to_csv("positive_scaffold_counts.csv", index=False)

# === 8. In thông báo ===
print(f"✅ Đã xử lý {len(df_data)} compound.")
print(f"✅ Có {len(df_filtered)} compound khớp với top 10 scaffold dương.")
print(f"📊 Thống kê: positive_scaffold_counts.csv")
