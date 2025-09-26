from vina import Vina
from openbabel import openbabel
import pandas as pd
import autodock as ad
import autodock_lowest as al
import subprocess
import os

# Đọc dữ liệu
df = pd.read_csv(os.path.join("Mols", "NPASS_Ligand_20.csv"))

# Tạo thư mục Ligands
mol_filepath = os.path.join("Mols", "Ligands")
os.makedirs(mol_filepath, exist_ok=True)

# Convert SMILES -> MOL -> PDBQT
for index, row in df.iterrows():
    smiles = row['canonical_smiles']
    ligand_id = row['LigandID']

    mol_file = os.path.join(mol_filepath, f"{ligand_id}.mol")
    pdbqt_file = os.path.join(mol_filepath, f"{ligand_id}.pdbqt")

    ad.convert_smiles_mol(smiles, mol_file)
    print(f"[OK] Converted SMILES -> MOL for {ligand_id}")

    ad.change_mol_pdbqt(mol_file, pdbqt_file)
    print(f"[OK] Converted MOL -> PDBQT for {ligand_id}")

# Docking
for ligand_id in df['LigandID']:
    print(f"Processing ligand ID: {ligand_id}")

    receptor_path = "/home/andy/andy/Inflam_NP/molecular_docking/Protein_clean"
    input_receptor = "COX1_1EQH"
    ligand_path = mol_filepath
    output_filepath = os.path.join("InFlam_dock", "autodock")
    output_txt = os.path.join("InFlam_dock", "txt")

    os.makedirs(output_filepath, exist_ok=True)
    os.makedirs(output_txt, exist_ok=True)

    # Define the grid
    center_x, center_y, center_z = 47.46, 27.948, 193.164
    search_x, search_y, search_z = 35.0, 35.0, 35.0

    output_file = os.path.join(output_txt, f"{ligand_id}.txt")

    try:
        ad.vina_run(
            receptor_path, ligand_path, output_filepath, 
            input_receptor, ligand_id,
            center_x, center_y, center_z, search_x, search_y, search_z
        )

        # Log output
        command = (
            f"from autodock import vina_run; "
            f"vina_run('{receptor_path}', '{ligand_path}', '{output_filepath}', "
            f"'{input_receptor}', '{ligand_id}', "
            f"{center_x}, {center_y}, {center_z}, {search_x}, {search_y}, {search_z})"
        )
        with open(output_file, "w") as outfile:
            subprocess.run(
                ["python", "-c", command],
                stdout=outfile,
                stderr=subprocess.STDOUT
            )

        print(f"[OK] Docking completed for {ligand_id}")

    except RuntimeError as e:
        if "outside the grid box" in str(e):
            print(f"Skipping {ligand_id}: {e}")
        else:
            print(f"Error {ligand_id}: {e}")
        continue

    except Exception as e:
        print(f"Unexpected error {ligand_id}: {e}")
        continue

# Save Lowest Affinity
print("Processing lowest affinity...")

ligand_ids = df['LigandID'].dropna().tolist()
txt_path = os.path.join("InFlam_dock", "txt")
lowest_csv = os.path.join("InFlam_dock", "lowest_affinity.csv")

affinities = al.read_ligand(ligand_ids, txt_path)
al.find_lowest(ligand_ids, affinities, lowest_csv)
