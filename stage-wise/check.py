from rdkit import Chem
import sascorer
import traceback

smiles = 'Nc1c(C(=O)N2CCOCC2)sc2nc3c(cc12)CCCC3'
mol = Chem.MolFromSmiles(smiles)

try:
    score = sascorer.calculateScore(mol)
    print(f"Test Score: {score}")
except Exception:
    print("SA Scorer đang bị lỗi, chi tiết:")
    traceback.print_exc()