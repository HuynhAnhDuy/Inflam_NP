import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# Đọc file CSV
df = pd.read_csv('3.InFlamNat_preprocess.csv')  # thay thế bằng đường dẫn thực tế

# Giả sử cột SMILES tên là 'canonical_smiles'
def get_murcko_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    else:
        return None

# Tạo cột mới chứa Murcko scaffold
df['murcko_scaffold'] = df['canonical_smiles'].apply(get_murcko_scaffold)

# Lưu kết quả ra file mới (tuỳ chọn)
df.to_csv('3.InFlamNat_preprocess_scaffold.csv', index=False)
