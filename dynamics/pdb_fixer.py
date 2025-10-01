from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
from pdbfixer import PDBFixer
import os


def pdb_fixer(input_file, output_file):
    fixer = PDBFixer(filename=input_file)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(True)   # ⚠️ sẽ xóa cả ligand nếu có
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    # Nếu không có unit cell trong PDB thì dùng padding
    if fixer.topology.getUnitCellDimensions() is not None:
        fixer.addSolvent(fixer.topology.getUnitCellDimensions())
    else:
        fixer.addSolvent(padding=1.0*nanometer, ionicStrength=0.15*molar)

    PDBFile.writeFile(fixer.topology, fixer.positions, open(output_file, 'w'))
    print(f"✅ Successfully fixed PDB file!\nFixed PDB saved to {output_file}")


def main():
    input_molecule = "5LOX_6NCF_clean"
    input_file = input_molecule + ".pdb"
    output_file = input_molecule + "_fixed.pdb"
    pdb_fixer(input_file, output_file)


if __name__ == "__main__":
    main()
