import os
import pandas as pd
from openbabel import openbabel

# ==========================
# Function helpers
# ==========================
def change_pdb_pdbqt(input_file, output_file):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdb", "pdbqt")
    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, input_file)
    obConversion.WriteFile(mol, output_file)
    print(f"[OK] Receptor saved: {output_file}")

def clean_pdbqt(input_path, output_path):
    with open(input_path, 'r') as infile:
        lines = infile.readlines()
    with open(output_path, 'w') as outfile:
        for line in lines:
            if not line.startswith(('ROOT','ENDROOT','BRANCH','ENDBRANCH','TORSDOF')):
                outfile.write(line)
    print(f"[OK] Clean receptor saved: {output_path}")

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
    receptor_dir = "Protein_clean"     # chứa file *_clean.pdb
    ligand_dir = "ligands"
    os.makedirs(ligand_dir, exist_ok=True)

    # ==== Xử lý toàn bộ protein trong folder ====
    pdb_files = [f for f in os.listdir(receptor_dir) if f.endswith("_clean.pdb")]
    for pdb_file in pdb_files:
        receptor_name = os.path.splitext(pdb_file)[0]  # bỏ .pdb
        pdb_path = os.path.join(receptor_dir, pdb_file)
        pdbqt_path = os.path.join(receptor_dir, receptor_name + ".pdbqt")
        receptor_clean = os.path.join(receptor_dir, receptor_name + "_clean.pdbqt")

        change_pdb_pdbqt(pdb_path, pdbqt_path)
        clean_pdbqt(pdbqt_path, receptor_clean)

    # ==== Xử lý ligands từ CSV ====
    df = pd.read_csv("NPASS_Ligand_20.csv")
    for i, smiles in enumerate(df["canonical_smiles"], start=1):
        ligand_name = f"Ligand_{i}"
        mol_file = os.path.join(ligand_dir, ligand_name + ".mol")
        pdbqt_file = os.path.join(ligand_dir, ligand_name + ".pdbqt")

        convert_smiles_mol(smiles, mol_file)
        change_mol_pdbqt(mol_file, pdbqt_file)

if __name__ == "__main__":
    main()
