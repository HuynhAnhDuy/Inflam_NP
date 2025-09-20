import pandas as pd
from rdkit import Chem

# === CONFIG ===
THRESHOLD = 0.8   # dễ điều chỉnh

# === Load dữ liệu ===
df_maccs = pd.read_csv("NPASS_scaffold_hopping_topN_maccs.csv")
df_rdkit = pd.read_csv("NPASS_scaffold_hopping_topN_rdkit.csv")

# === Giữ các cột cần thiết ===
cols = ["canonical_smiles1", "canonical_smiles2", "similarity"]
df_maccs = df_maccs[cols].copy()
df_rdkit = df_rdkit[cols].copy()

# === Lọc theo similarity > THRESHOLD ===
df_maccs = df_maccs[df_maccs["similarity"] > THRESHOLD]
df_rdkit = df_rdkit[df_rdkit["similarity"] > THRESHOLD]

# === Tạo khóa định danh cặp (đảm bảo thứ tự không quan trọng) ===
df_maccs["pair_key"] = df_maccs.apply(
    lambda x: tuple(sorted([x["canonical_smiles1"], x["canonical_smiles2"]])), axis=1
)
df_rdkit["pair_key"] = df_rdkit.apply(
    lambda x: tuple(sorted([x["canonical_smiles1"], x["canonical_smiles2"]])), axis=1
)

# === Tìm giao nhau ===
common_pairs = pd.merge(
    df_maccs, df_rdkit,
    on="pair_key",
    suffixes=("_maccs", "_rdkit")
)

# === Thêm cột similarity_mean ===
common_pairs["similarity_mean"] = (
    common_pairs["similarity_maccs"] + common_pairs["similarity_rdkit"]
) / 2

# === Hàm chuẩn hóa SMILES bằng RDKit ===
def to_canonical(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

# === Sinh SMILES chuẩn cho từng cột (trên common_pairs) ===
common_pairs["smiles1_can"] = common_pairs["canonical_smiles1_maccs"].apply(to_canonical)
common_pairs["smiles2_can"] = common_pairs["canonical_smiles2_maccs"].apply(to_canonical)

# === Lọc cặp trùng phân tử (dù SMILES khác nhau) ===
df_same_molecule = common_pairs[common_pairs["smiles1_can"] == common_pairs["smiles2_can"]]

# === Sort theo similarity_mean giảm dần ===
common_pairs = common_pairs.sort_values(by="similarity_mean", ascending=False)

# === Xuất kết quả ===
common_pairs.to_csv("NPASS_common_scaffold_hopping.csv", index=False)
df_same_molecule.to_csv("pairs_same_molecule.csv", index=False)

print(f"✅ Tìm được {len(common_pairs)} cặp chung có similarity > {THRESHOLD}")
print(f"✅ Trong đó có {len(df_same_molecule)} cặp mà canonical_smiles1 và canonical_smiles2 thực chất là cùng phân tử")
print("📂 Kết quả đã lưu trong 'common_scaffold_hopping.csv' (sort theo similarity_mean giảm dần)")
