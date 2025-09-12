import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

# === STEP 1: Đọc file enrichment summary ===
df = pd.read_csv("/home/andy/andy/Inflam_NP/R_group_identification/rgroup_shap_only_active.csv")

# Lọc bỏ những dòng rỗng
df = df.dropna(subset=["rgroup_smiles"])

# === STEP 2: Lấy danh sách R-group duy nhất ===
# Giữ lại rgroup_smiles, active_count, inactive_count
df_unique = df.drop_duplicates(subset=["rgroup_smiles"])[
    ["rgroup_smiles", "active_count", "inactive_count"]
]
rgroups = df_unique["rgroup_smiles"].tolist()
print("👉 Tổng số R-group duy nhất:", len(rgroups))

# === STEP 3: Chuyển SMILES thành Mol + tạo legend ===
mols = []
legends = []

for _, row in df_unique.iterrows():
    smi = row["rgroup_smiles"]
    mol = Chem.MolFromSmiles(smi)
    if mol:
        mols.append(mol)
        legend = f"{smi}\nAct={row['active_count']} | Inact={row['inactive_count']}"
        legends.append(legend)

# === STEP 4: Vẽ toàn bộ R-group dạng grid và xuất SVG ===
svg = Draw.MolsToGridImage(
    mols,
    molsPerRow=5,           # tăng số cột để giảm chiều cao
    subImgSize=(300, 300),  # giảm kích thước mỗi ô
    legends=legends,
    useSVG=True             # xuất SVG thay vì PNG
)

with open("all_rgroups_only_active.svg", "w") as f:
    f.write(svg)

print("✅ Đã lưu toàn bộ R-group với số lượng Active/Inactive dưới dạng SVG (hàng gọn hơn)")
