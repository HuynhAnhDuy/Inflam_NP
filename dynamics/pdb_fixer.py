from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
from pdbfixer import PDBFixer
from openmm.app import PDBFile
import os


def pdb_fixer(input_file,output_file):
    '''
    This function fix problems in PDB files in preparation to simulate the molecular
    ------
    Parameters:
    input_file: original PDB file
    output_file: fixed PDB file
    '''
    fixer = PDBFixer(filename=input_file)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(True)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    fixer.addSolvent(fixer.topology.getUnitCellDimensions())
    PDBFile.writeFile(fixer.topology, fixer.positions, open(output_file, 'w'))
    return print(f'''Successfully fixed PDB file!
    Fixed PDB file saved to {output_file}''')
    

def main():
    '''
    This is the main function to run all the functions above.
    This also contains input to insert the name of the molecule.
    --------
    pdb_fixer will run to fix the PDB files.
    run_openmm will run to calculate the force field and give the output result.
    '''
    # Input file to fix PDB file
    input_molecule = input("Please type your molecule name ")
    input_file = os.path.join(input_molecule+'.pdb')
    output_file = os.path.join(input_molecule+'_fixed.pdb')
    pdb_fixer(input_file,output_file)
    
    
if "__name__" == "__main__":
       main()

main()
