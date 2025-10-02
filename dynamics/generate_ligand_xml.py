#!/usr/bin/env python3
"""
Generate OpenMM ligand XML from a ligand structure file using OpenFF Toolkit.
Input:  Ligand file (mol/sdf/mol2/pdb) with 3D coordinates
Output: ligand.xml
"""

from openff.toolkit.topology import Molecule
from openmmforcefields.generators import SystemGenerator

def main():
    # --- chỉnh trực tiếp tại đây ---
    input_file = "Ligand_5_out_5LOX_6NCF.mol"   # file input ligand
    output_file = "Ligand_5_out_5LOX_6NCF.xml"  # file output XML
    forcefield = "openff-2.0.0"                 # force field cho ligand
    # --------------------------------

    print(f"[Info] Reading ligand from {input_file}")
    ligand = Molecule.from_file(input_file)

    if ligand.n_conformers == 0:
        raise ValueError("Ligand has no 3D conformer. Please provide 3D coordinates.")

    print("[Info] Preparing SystemGenerator...")
    system_generator = SystemGenerator(
        small_molecule_forcefield=forcefield,
        molecules=[ligand]
    )

    ff = system_generator.createForceField()

    print(f"[Info] Writing ligand parameters to {output_file}")
    ff.writeFile(output_file)
    print("[Done] ligand.xml created successfully.")

if __name__ == "__main__":
    main()
