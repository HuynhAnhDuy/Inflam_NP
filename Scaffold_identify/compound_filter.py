import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# === 1. Đọc dữ liệu SMILES ===
df = pd.read_csv("capsule_x_train.csv")  # hoặc file nào bạn muốn tìm trong

# === 2. Hàm lấy scaffold ===
def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    return None

# === 3. Tạo cột scaffold nếu chưa có ===
if 'scaffold' not in df.columns:
    df['scaffold'] = df['canonical_smiles'].apply(get_scaffold)

# === 4. Lọc theo scaffold cụ thể ===
target_scaffold = "C1=Cc2ccccc2/C1=C\c1ccccc1"
df_match = df[df['scaffold'] == target_scaffold]

# === 5. Xuất kết quả ra file hoặc in ra ===
df_match.to_csv("matched_Indenyl-vinyl-phenyl_compounds.csv", index=False)
print(f"✅ Found {len(df_match)} compounds containing the target scaffold.")
print(df_match[['canonical_smiles', 'Toxicity Value']].head())
