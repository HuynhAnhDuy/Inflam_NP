import pandas as pd 
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import fisher_exact
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# === 1) Đọc dữ liệu ===
input_file = "3.InFlamNat_preprocess.csv"   # 👈 thay bằng tên file của bạn
df = pd.read_csv(input_file)

# --- Chuẩn hoá tên cột ---
if 'SMILES' in df.columns and 'canonical_smiles' not in df.columns:
    df = df.rename(columns={'SMILES': 'canonical_smiles'})

required_cols = {'canonical_smiles', 'Label'}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Thiếu cột bắt buộc: {missing}. "
                     f"Hãy chắc rằng file có cột 'canonical_smiles' và 'Label' (0/1).")

if not np.issubdtype(df['Label'].dtype, np.integer):
    df['Label'] = df['Label'].map({1:1, 0:0, '1':1, '0':0, True:1, False:0, 'active':1, 'inactive':0})
df['Label'] = df['Label'].fillna(0).astype(int)
df = df[df['Label'].isin([0,1])].copy()

df['canonical_smiles'] = df['canonical_smiles'].astype(str).str.strip()
df = df[df['canonical_smiles'] != ""].copy()

# === 2) Trích xuất scaffold ===
def get_scaffold(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, isomericSmiles=True)
    except Exception:
        return None

df['Scaffold'] = df['canonical_smiles'].apply(get_scaffold)
df = df.dropna(subset=['Scaffold']).copy()

# === 3) Đếm tần suất theo Label ===
pos_count = df.loc[df['Label'] == 1, 'Scaffold'].value_counts()
neg_count = df.loc[df['Label'] == 0, 'Scaffold'].value_counts()

scaffold_df = pd.DataFrame({'Pos': pos_count, 'Neg': neg_count}).fillna(0).astype(int)

# Tổng số mẫu theo nhóm
total_pos = int((df['Label'] == 1).sum())
total_neg = int((df['Label'] == 0).sum())

# === 4) Fisher's Exact Test (Haldane–Anscombe) ===
def fisher_with_correction(row):
    a = int(row['Pos'])
    b = int(row['Neg'])
    c = total_pos - a
    d = total_neg - b
    table = np.array([[a, b], [c, d]], dtype=float)
    if (table == 0).any():
        table = table + 0.5
    oddsratio, p = fisher_exact(table, alternative='two-sided')
    return pd.Series({'OddsRatio': oddsratio, 'p_value': p})

stats = scaffold_df.apply(fisher_with_correction, axis=1)
scaffold_df = scaffold_df.join(stats)

# Tần suất & Enrichment
scaffold_df['Pos_freq'] = scaffold_df['Pos'] / max(total_pos, 1)
scaffold_df['Neg_freq'] = scaffold_df['Neg'] / max(total_neg, 1)
scaffold_df['Enrichment'] = (scaffold_df['Pos_freq'] / scaffold_df['Neg_freq']).replace([np.inf, -np.inf], np.nan)
scaffold_df['log2_Enrichment'] = np.log2(scaffold_df['Enrichment'])

# === 5) Đánh dấu Significant ===
scaffold_df['Significant'] = (
    (scaffold_df['p_value'] < 0.05) &
    (scaffold_df['Enrichment'] > 2) )

# === 6) Xuất full kết quả ===
scaffold_df = scaffold_df.sort_values(by='p_value', ascending=True)
full_out = "3.InFlamNat_preprocess_scaffold_statistic.csv"
scaffold_df.to_csv(full_out, index=True)

# === 7) Xuất danh sách scaffold Significant ===
sig_pos_only = scaffold_df[scaffold_df['Significant']].copy()
sig_pos_only = sig_pos_only.sort_values(['p_value','Pos'], ascending=[True, False])

pos_out = "3.InFlamNat_preprocess_scaffold_POSonly_sig.csv"
sig_pos_only.to_csv(pos_out, index=True)

print(f"✅ Phân tích xong.")
print(f" - Full (có cột Significant): {full_out}")
print(f" - POS-only significant: {pos_out}")

print("Tổng scaffold:", len(scaffold_df))
print("p<0.05:", (scaffold_df['p_value'] < 0.05).sum())
print("Enrichment>2", (scaffold_df['Enrichment'] > 2).sum())
