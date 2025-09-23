from vina import Vina
from openbabel import openbabel
import pandas as pd
import autodock as ad
import autodock_lowest as al
import subprocess
import os


df = pd.read_csv(os.path.join('ampk_dock','ampk_compounds.csv'))

# Convert SMILES to MOL to PDBQT
for index, row in df.iterrows():
     smiles = row['canonical_smiles']
     ligand_id = row['LigandID']
     
     mol_filepath = os.path.join('ampk_dock', 'mols')
     
     # convert smiles to mol
     mol_file = os.path.join(mol_filepath, f'{ligand_id}.mol')
     ad.convert_smiles_mol(smiles,mol_file)
     print(f"Successfully converted {ligand_id} at index {index}")

     # convert mol to pdbqt
     mol_file = os.path.join(mol_filepath, f'{ligand_id}.mol')
     pdbqt_file = os.path.join(mol_filepath, f'{ligand_id}.pdbqt')
     ad.change_mol_pdbqt(mol_file,pdbqt_file)
     print(f"Successfully converted {ligand_id} at index {index}")


# Docking Loop
for index, ligand_id in df['LigandID'].items():
    
    print(f'Processing ligand ID: {ligand_id}')
    receptor_path = 'ampk_dock/protein'
    input_receptor = 'ampk_prot'
    ligand_path = os.path.join('ampk_dock','mols')
    input_ligand = ligand_id
    output_filepath = os.path.join('ampk_dock', 'autodock')
    output_txt = os.path.join('ampk_dock', 'txt')
    
    # Define the center and box size
    center_x, center_y, center_z = 89.18, -3597, 36.02
    search_x, search_y, search_z = 186.36, 158.43, 104.13
    
    # Prepare the output file for capturing terminal output
    output_file = os.path.join(output_txt, f"{ligand_id}.txt")
    
    try:
        # Run docking
        ad.vina_run(receptor_path, ligand_path, output_filepath, input_receptor, input_ligand,
                    center_x, center_y, center_z, search_x, search_y, search_z)
        
        # Capture terminal output using subprocess
        command = (
            f"from autodock import vina_run; "
            f"vina_run('{receptor_path}', '{ligand_path}', '{output_filepath}', '{input_receptor}', '{input_ligand}', "
            f"{center_x}, {center_y}, {center_z}, {search_x}, {search_y}, {search_z})"
        )
        
        with open(output_file, "w") as outfile:
            subprocess.run(
                ["python", "-c", command],
                stdout=outfile,
                stderr=subprocess.STDOUT
            )
        
        print(f"Docking completed for {ligand_id}.")
    
    except RuntimeError as e:
        if "The ligand is outside the grid box" in str(e):
            print(f"Skipping {ligand_id}: {e}")
        else:
            print(f"Error occurred for {ligand_id}: {e}")
        continue  # Skip to the next ligand if an error occurs
    
    except Exception as e:
        # Catch any other exceptions
        print(f"An unexpected error occurred for {ligand_id}: {e}")
        continue  # Continue to the next ligand

# Save Lowest Affinities
print("Processing to save the lowest affinity of each compound...")

# Read ligand IDs from the CSV file
ligand_ids = pd.read_csv(os.path.join('ampk_dock', 'ampk_compounds.csv'))['LigandID'].dropna().tolist()

# Directory paths
txt_path = 'ampk_dock/txt'
lowest_csv = os.path.join('ampk_dock', 'lowest_affinity.csv')

# Read affinities and find lowest
affinities = al.read_ligand(ligand_ids, txt_path)
al.find_lowest(ligand_ids, affinities, lowest_csv)

