#!/usr/bin/env python3
import os
import pandas as pd
from openbabel import openbabel

# ==========================
# Function helpers
# ==========================
def convert_smiles_mol(smiles, output_file):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("smi", "mol")
    mol = openbabel.OBMol()
    obConversion.ReadString(mol, smiles)
    mol.AddHydrogens()
    openbabel.OBBuilder().Build(mol)
    obConversion.WriteFile(mol, output_file)
    print(f"[OK] Ligand MOL saved: {output_file}")

def change_mol_pdbqt(input_file, output_file):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("mol", "pdbqt")
    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, input_file)
    obConversion.WriteFile(mol, output_file)
    print(f"[OK] Ligand PDBQT saved: {output_file}")

# ==========================
# Main workflow
# ==========================
def main():
    ligand_dir = "/home/andy/andy/Inflam_NP/molecular_docking/Controls"
    os.makedirs(ligand_dir, exist_ok=True)

    df = pd.read_csv("/home/andy/andy/Inflam_NP/molecular_docking/grid_centers_double_check.csv",encoding="latin-1")
    for i, smiles in enumerate(df["canonical_smiles"], start=1):
        ligand_name = f"Ligand_{i}_control"
        mol_file = os.path.join(ligand_dir, ligand_name + ".mol")
        pdbqt_file = os.path.join(ligand_dir, ligand_name + ".pdbqt")

        convert_smiles_mol(smiles, mol_file)
        change_mol_pdbqt(mol_file, pdbqt_file)

if __name__ == "__main__":
    main()
