"""
Run a MD simulation for a complex, optionally adding a solvent box
"""

import sys, time, argparse
from openff.toolkit import Molecule
from openmmforcefields.generators import SystemGenerator
import openmm
from openmm import app, unit, LangevinIntegrator, Vec3
from openmm.app import PDBFile, Simulation, Modeller, PDBReporter, StateDataReporter, DCDReporter
import argparse
import os
import subprocess
from openmm import Platform


def input_argument():
    parser = argparse.ArgumentParser(description="Molecular simulation script")
    # Set default values so they are not required to be input every time
    parser.add_argument("-p", "--protein", default=os.path.join('openmm',"2AZ5_nomolecule_fixed.pdb"), help="Protein PDB file")
    parser.add_argument("-l", "--ligand", default=os.path.join('mols',"mitragynine_rotate.mol"), help="Ligand molfile")
    parser.add_argument("-o", "--output", default=os.path.join('outputs',"complex_mitragynine_rotate_2AZ5"), help="Base name for output files")
    
    # Duration vs steps
    parser.add_argument("-s", "--steps", type=int, help="Number of steps (overrides duration if set)")  #default=500000
    parser.add_argument("--duration-ns", type=float, default=1.0, help="Target simulation duration in ns")
    
    parser.add_argument("-z", "--step-size", type=float, default=0.002, help="Step size (ps)")
    parser.add_argument("-f", "--friction-coeff", type=float, default=1.0, help="Friction coefficient (ps)")
    parser.add_argument("-i", "--interval", type=int, default=500, help="Reporting interval")   #500 (saves disk space_)
    parser.add_argument("-t", "--temperature", type=int, default=300, help="Temperature (K)")
    parser.add_argument("--solvate", action='store_true', default=True, help="Add solvent box")
    parser.add_argument("--padding", type=float, default=1.0, help="Padding for solvent box (Å)")
    parser.add_argument("--water-model", default="tip3p",
                        choices=["tip3p", "spce", "tip4pew", "tip5p", "swm4ndp"],
                        help="Water model for solvation")
    parser.add_argument("--positive-ion", default="Na+", help="Positive ion for solvation")
    parser.add_argument("--negative-ion", default="Cl-", help="Negative ion for solvation")
    parser.add_argument("--ionic-strength", type=float, default=0.0, help="Ionic strength for solvation")
    parser.add_argument("--no-neutralize", action='store_true', default=False, help="Don't add ions to neutralize")
    parser.add_argument("-e", "--equilibration-steps", type=int, default=200, help="Number of equilibration steps")
    parser.add_argument("--protein-force-field", default='amber/ff14SB.xml', help="Protein force field")
    parser.add_argument("--ligand-force-field", default='gaff-2.11', help="Ligand force field")
    parser.add_argument("--water-force-field", default='amber/tip3p_standard.xml', help="Water force field")

    args = parser.parse_args()

    # If steps not provided, calculate from duration and step size
    if args.steps is None:
        total_ps = args.duration_ns * 1000.0  # ns -> ps
        args.steps = int(total_ps / args.step_size)

    return args
 

def create_complex(args,mol_in,pdb_in,output_complex):
    # get the chosen or fastest platform
    platform = Platform.getPlatformByName('CPU')

    print('Reading ligand')
    ligand_mol = Molecule.from_file(mol_in)

    print('Preparing system')
    # Initialize a SystemGenerator using the GAFF for the ligand and tip3p for the water.
    forcefield_kwargs = {'constraints': app.HBonds, 'rigidWater': True, 'removeCMMotion': False, 'hydrogenMass': 4*unit.amu }
    system_generator = SystemGenerator(
        forcefields=[args.protein_force_field, args.water_force_field],
        small_molecule_forcefield=args.ligand_force_field,
        molecules=[ligand_mol],
        forcefield_kwargs=forcefield_kwargs)

    # Use Modeller to combine the protein and ligand into a complex
    print('Reading protein')
    protein_pdb = PDBFile(pdb_in)

    print('Preparing complex')
    # The topology is described in the openforcefield API

    modeller = Modeller(protein_pdb.topology, protein_pdb.positions)
    print('System has %d atoms' % modeller.topology.getNumAtoms())

    # The topology is described in the openforcefield API
    print('Adding ligand...')
    lig_top = ligand_mol.to_topology()
    modeller.add(lig_top.to_openmm(), lig_top.get_positions().to_openmm())
    print('System has %d atoms' % modeller.topology.getNumAtoms())

    # Solvate
    if args.solvate:
        print('Adding solvent...')
        # we use the 'padding' option to define the periodic box.
        # we just create a box that has a 10A (default) padding around the complex.
        modeller.addSolvent(system_generator.forcefield, model=args.water_model, padding=args.padding * unit.angstroms,
                            positiveIon=args.positive_ion, negativeIon=args.negative_ion,
                            ionicStrength=args.ionic_strength * unit.molar, neutralize=not args.no_neutralize)
        print('System has %d atoms' % modeller.topology.getNumAtoms())

    with open(output_complex, 'w') as outfile:
        PDBFile.writeFile(modeller.topology, modeller.positions, outfile)
        
    return platform, ligand_mol, modeller, system_generator 
    
    

def create_system(platform,modeller, system_generator, args, ligand_mol, num_steps, temperature, equilibration_steps, output_min):
    # Create the system using the SystemGenerator
    system = system_generator.create_system(modeller.topology, molecules=ligand_mol)

    friction_coeff = args.friction_coeff / unit.picosecond
    step_size = args.step_size * unit.picoseconds
    duration = (step_size * num_steps).value_in_unit(unit.nanoseconds)
    print('Simulating for {} ns'.format(duration))

    integrator = LangevinIntegrator(temperature, friction_coeff, step_size)
    if args.solvate:
        system.addForce(openmm.MonteCarloBarostat(1 * unit.atmospheres, temperature, 25))

    if system.usesPeriodicBoundaryConditions():
        print('Default Periodic box: {}'.format(system.getDefaultPeriodicBoxVectors()))
    else:
        print('No Periodic Box')

    simulation = Simulation(modeller.topology, system, integrator, platform=platform)
    context = simulation.context
    context.setPositions(modeller.positions)

    print('Minimising ...')
    simulation.minimizeEnergy()

    # Write out the minimised PDB.
    with open(output_min, 'w') as outfile:
        PDBFile.writeFile(modeller.topology, context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(), file=outfile, keepIds=True)

    # equilibrate
    simulation.context.setVelocitiesToTemperature(temperature)
    print('Equilibrating ...')
    simulation.step(equilibration_steps)
    return simulation,duration

def openmm_run():
    t0 = time.time()
    
    args = input_argument()
    
    pdb_in = args.protein
    mol_in = args.ligand
    output_base = args.output
    output_complex = output_base + '_complex.pdb'
    output_traj_dcd = output_base + '_traj.dcd'
    output_min = output_base + '_minimised.pdb'
    num_steps = args.steps
    reporting_interval = args.interval
    temperature = args.temperature * unit.kelvin
    equilibration_steps = args.equilibration_steps
    print('Processing', pdb_in, 'and', mol_in, 'with', num_steps, 'steps generating outputs',
          output_complex, output_min, output_traj_dcd)
    
    platform, ligand_mol, modeller, system_generator =create_complex(args,mol_in,pdb_in,output_complex)
    simulation,duration=create_system(platform,modeller, system_generator, args, ligand_mol, num_steps, temperature, equilibration_steps, output_min)
    
    # Run the simulation.
    simulation.reporters.append(PDBReporter(output_base + '_traj.pdb', reporting_interval)) #new
    simulation.reporters.append(DCDReporter(output_traj_dcd, reporting_interval, enforcePeriodicBox=True))
    simulation.reporters.append(StateDataReporter(sys.stdout, reporting_interval * 5, step=True, potentialEnergy=True, temperature=True))
    print('Starting simulation with', num_steps, 'steps ...')
    
    t1 = time.time()
    simulation.step(num_steps)
    t2 = time.time()
    print('Simulation complete in {} mins at {}. Total wall clock time was {} mins'.format(
        round((t2 - t1) / 60, 3), temperature, round((t2 - t0) / 60, 3)))
    print('Simulation time was', round(duration, 3), 'ns')
    
    return "Successfully done" #+ str(i)

openmm_run()