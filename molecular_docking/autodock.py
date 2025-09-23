from vina import Vina
from openbabel import openbabel
from rdkit import Chem
import os

def change_pdb_pdbqt(input_file,output_file):
    '''
    This function load pdb and turn to pdbqt file. It will save the pdbqt file into the output path.
    -------
    Parameters:
    Input_file: PDB file path
    Output_file: PDBQT file path
    '''
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdb", "pdbqt")
    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, input_file)
    obConversion.WriteFile(mol, output_file)
    return print(f"Successfully changed PDB to PDBQT! File saved to {output_file}")

def clean_pdbqt(input_path, output_path):
    '''
    This function is to clean only the receptor in pdbqt file.
    --------
    Parameters:
    Input_path: receptor before its cleaned
    Output_path: receptor after its cleaned
    --------
    Get rid of ROOT, ENDROOT, BRANCH, ENDBRANCH, TORSDOF, so it can run on Autodock Vina
    '''
    with open(input_path, 'r') as infile:
        lines = infile.readlines()
    
    with open(output_path, 'w') as outfile:
        for line in lines:
            if not line.startswith(('ROOT', 'ENDROOT', 'BRANCH', 'ENDBRANCH', 'TORSDOF')):
                outfile.write(line)
    return print(f"Successfully cleaned PDBQT!  File saved to {output_path}")

def convert_smiles_mol(smiles,output_file):
    '''
    This function is to convert SMILES strings to MOL file.
    The oputput will save to the output file path.
    -----
    Parameters:
    input_smiles: SMILES strings
    output_file: MOL file path
    '''
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("smi", "mol")
    
    mol = openbabel.OBMol()
    obConversion.ReadString(mol,smiles)
    mol.AddHydrogens()
    openbabel.OBBuilder().Build(mol) # To get coordinates
    mol2string = obConversion.WriteString(mol)  # If you just want the string, print it
    
    # To save as a .mol file
    obConversion.WriteFile(mol, output_file)
    return print(f"Successfully changed SMILES to MOL! File saved to {output_file}")

def change_mol_pdbqt(input_file,output_file):
    '''
    This function load pdb and turn to pdbqt file. It will save the pdbqt file into the output path.
    -------
    Parameters:
    Input_file: MOL file path
    Output_file: PDBQT file path
    '''
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("mol", "pdbqt")
    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, input_file)
    obConversion.WriteFile(mol, output_file)
    return print(f"Successfully changed MOL to PDBQT! File saved to {output_file}")

def vina_run(receptor_path, ligand_path, output_filepath,input_receptor,input_ligand,center_x, center_y, center_z, search_x, search_y, search_z):
    '''
    This function is to run Autodock Vina. It will save the best 5 poses of the ligand.
    -------
    Parameters:
    input_receptor: receptor file name
    input_ligand: ligand file name
    
    center_x: x of the center
    center_y: y of the center
    center_z: z of the center
    
    search_x: x of the dimension
    search_y: y of the dimension
    search_z: z of the dimension
    '''
    
    v = Vina(sf_name='vina')
    v.set_receptor(os.path.join(receptor_path, input_receptor + '_clean.pdbqt'))
    v.set_ligand_from_file(os.path.join(ligand_path, input_ligand + '.pdbqt'))
    v.compute_vina_maps(center=[center_x, center_y, center_z], box_size=[search_x, search_y, search_z])
    energy = v.score()
    print('Score before minimization: %.3f (kcal/mol)' % energy[0])
    
    # Minimized locally the current pose
    energy_minimized = v.optimize()
    print('Score after minimization : %.3f (kcal/mol)' % energy_minimized[0])
    v.write_pose(os.path.join(output_filepath,input_receptor+'_'+input_ligand + '_minimized.pdbqt'), overwrite=True)

    # Dock the ligand
    v.dock(exhaustiveness=32, n_poses=5)
    poses_path = os.path.join(output_filepath,input_receptor+'_'+input_ligand + '_vina_out.pdbqt')
    v.write_poses(poses_path, n_poses=5, overwrite=True)
    
    
    return print("Successfully done AUTODOCK VINA")


def main():
    ''''
    This is the main function for change pdb to pdbqt, change mol to pdbqt, clean pdbqt.
    This will run all the function including vina_run.
    '''
    
    # pdb to pdbqt
    input_receptor = input("Please type your receptor name ")
    input_file = os.path.join(input_path, input_receptor+'.pdb')      # Replace with your input file path and file name
    output_file = os.path.join(input_path, input_receptor+'.pdbqt')   # Replace with your desired output file path
    change_pdb_pdbqt(input_file, output_file)
    
    #clean pdbqt
    input_path = os.path.join('mols',input_receptor+'.pdbqt')           
    output_path = os.path.join('mols',input_receptor+'_clean.pdbqt')     
    clean_pdbqt(input_path, output_path)
    
    #smiles to mol
    smiles = input("Please type your SMILES strings ")
    input_name = input("Please type your desire output file name ")
    output_file = os.path.join('mols', input_name+'.mol')
    convert_smiles_mol(smiles,output_file)
    
    # mol to pdbqt
    input_ligand = input("Please type your ligand name ")
    input_file = os.path.join('mols', input_ligand+'.mol')      # Replace with your input file path and file name
    output_file = os.path.join('mols', input_ligand+'.pdbqt')   # Replace with your desired output file path
    change_mol_pdbqt(input_file, output_file)
    
    # insert center and box_size
    center_x = float(input("Please type your center x "))
    center_y = float(input("Please type your center y "))
    center_z = float(input("Please type your center z "))
    
    search_x = float(input("Please type your dimension x "))
    search_y = float(input("Please type your dimension y "))
    search_z = float(input("Please type your dimension z "))
    
    # vina_run
    output_filepath='output'
    receptor_path='mols'
    ligand_path='mols/train/test'
    vina_run(receptor_path, ligand_path, output_filepath,input_receptor,input_ligand,center_x, center_y, center_z, search_x, search_y, search_z)
    
if "__name__" == "__main__":
       main()








