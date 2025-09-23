from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
import pandas as pd
from typing import Optional

# ==== CONFIG: chỉnh trực tiếp ở đây ====
INPUT_CSV  = "/home/andy/andy/Inflam_NP/Scaffold_identify/InFlam_full.csv"        # CSV input, phải có cột canonical_smiles
OUTPUT_CSV = "InFlam_full_with_scaffolds.csv"   # CSV output sau khi thêm scaffold
SMILES_COL = "canonical_smiles"            # Tên cột chứa SMILES
# =======================================

# === Chuẩn hoá phân tử trước khi lấy scaffold ===
def _standardize_mol(mol: Chem.Mol) -> Optional[Chem.Mol]:
    if mol is None:
        return None
    try:
        params = rdMolStandardize.CleanupParameters()
        mol = rdMolStandardize.Cleanup(mol, params)
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)     # giữ mảnh lớn nhất
        mol = rdMolStandardize.Uncharger().uncharge(mol)                # trung hoá điện tích
        mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)   # canonical tautomer
        return mol
    except Exception:
        return None

# === Lấy scaffold gốc từ SMILES ===
def murcko_from_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    mol = _standardize_mol(mol)
    if mol is None:
        return None
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        if core is None or core.GetNumAtoms() == 0:
            return None
        return Chem.MolToSmiles(core, isomericSmiles=False, kekuleSmiles=False, canonical=True)
    except Exception:
        return None

# === Thêm scaffold gốc vào DataFrame ===
def add_murcko_scaffolds_to_df(df: pd.DataFrame,
                               smiles_col: str = "canonical_smiles",
                               scaffold_col: str = "scaffold") -> pd.DataFrame:
    assert smiles_col in df.columns, f"Không thấy cột '{smiles_col}' trong CSV."
    df[scaffold_col] = df[smiles_col].astype(str).apply(murcko_from_smiles)
    return df

def main():
    df = pd.read_csv(INPUT_CSV)
    df_out = add_murcko_scaffolds_to_df(df, smiles_col=SMILES_COL, scaffold_col="scaffold")
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Đã xử lý xong. Kết quả lưu tại: {OUTPUT_CSV}")
    print(f"📌 Cột thêm: scaffold")

if __name__ == "__main__":
    main()
