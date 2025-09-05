from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
import pandas as pd

# ==== CONFIG: chỉnh trực tiếp ở đây ====
INPUT_CSV  = "3.InFlamNat_SHAP.csv"        # CSV input, phải có cột canonical_smiles
OUTPUT_CSV = "3.InFlamNat_SHAP_with_scaffolds.csv"          # CSV output sau khi thêm scaffold
SMILES_COL = "canonical_smiles"            # Tên cột chứa SMILES
# =======================================

# === Chuẩn hoá phân tử trước khi lấy scaffold ===
def _standardize_mol(mol: Chem.Mol):
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

def _murcko_from_smiles(smiles: str, generic: bool = False) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    mol = _standardize_mol(mol)
    if mol is None:
        return None
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        if core is None or core.GetNumAtoms() == 0:
            return None
        if generic:
            core = MurckoScaffold.MakeScaffoldGeneric(core)
        return Chem.MolToSmiles(core, isomericSmiles=False, kekuleSmiles=False, canonical=True)
    except Exception:
        return None

def add_murcko_scaffolds_to_df(df: pd.DataFrame,
                               smiles_col: str = "canonical_smiles",
                               exact_col: str = "scaffold",
                               generic_col: str = "scaffold_generic") -> pd.DataFrame:
    assert smiles_col in df.columns, f"Không thấy cột '{smiles_col}' trong CSV."
    smi_series = df[smiles_col].astype(str)

    df[exact_col]   = [ _murcko_from_smiles(smi, generic=False) for smi in smi_series ]
    df[generic_col] = [ _murcko_from_smiles(smi, generic=True)  for smi in smi_series ]

    return df

def main():
    df = pd.read_csv(INPUT_CSV)
    df_out = add_murcko_scaffolds_to_df(df, smiles_col=SMILES_COL,
                                        exact_col="scaffold", generic_col="scaffold_generic")
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Đã xử lý xong. Kết quả lưu tại: {OUTPUT_CSV}")
    print(f"📌 Cột thêm: scaffold, scaffold_generic")

if __name__ == "__main__":
    main()
