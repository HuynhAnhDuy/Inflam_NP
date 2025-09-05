import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

def get_murcko_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None or scaf.GetNumAtoms() == 0:
            return None
        # canonical SMILES nhất quán, bỏ stereo
        return Chem.MolToSmiles(scaf, isomericSmiles=False, kekuleSmiles=False, canonical=True)
    except Exception:
        return None

df = pd.read_csv("3.InFlamNat_preprocess.csv")
df["murcko_scaffold"] = df["canonical_smiles"].apply(get_murcko_scaffold)
df.to_csv("3.InFlamNat_preprocess_scaffold.csv", index=False)
