"""
Run a MD simulation for a complex with checkpointing and chunked execution
"""

import sys, time, argparse, os
from openff.toolkit import Molecule
from openmmforcefields.generators import SystemGenerator
import openmm
from openmm import app, unit, LangevinIntegrator
from openmm.app import PDBFile, Simulation, Modeller, PDBReporter, StateDataReporter, DCDReporter
from openmm import Platform
import subprocess


def input_argument():
    parser = argparse.ArgumentParser(description="Molecular simulation script with checkpointing")
    parser.add_argument("-p", "--protein", default=os.path.join('openmm', "2AZ5_nomolecule_fixed.pdb"), help="Protein PDB file")
    parser.add_argument("-l", "--ligand", default=os.path.join('mols', "mitragynine_rotate.mol"), help="Ligand molfile")
    parser.add_argument("-o", "--output", default=os.path.join('outputs', "complex_mitragynine_rotate_2AZ5"), help="Base name for output files")
    parser.add_argument("-s", "--steps", type=int, default=20000, help="Total number of steps (e.g. 1 ns at 2 fs)")
    parser.add_argument("--chunk-steps", type=int, default=10000, help="Steps per chunk (smaller runs to avoid memory issues)")
    parser.add_argument("-z", "--step-size", type=float, default=0.002, help="Step size (ps)")
    parser.add_argument("-f", "--friction-coeff", type=float, default=1.0, help="Friction coefficient (ps)")
    parser.add_argument("-i", "--interval", type=int, default=1000, help="Reporting interval")
    parser.add_argument("-t", "--temperature", type=int, default=300, help="Temperature (K)")
    parser.add_argument("--solvate", action='store_true', default=True, help="Add solvent box")
    parser.add_argument("--padding", type=float, default=1.0, help="Padding for solvent box (Å)")
    parser.add_argument("--water-model", default="tip3p", choices=["tip3p", "spce", "tip4pew", "tip5p", "swm4ndp"], help="Water model for solvation")
    parser.add_argument("--positive-ion", default="Na+", help="Positive ion for solvation")
    parser.add_argument("--negative-ion", default="Cl-", help="Negative ion for solvation")
    parser.add_argument("--ionic-strength", type=float, default=0.0, help="Ionic strength for solvation")
    parser.add_argument("--no-neutralize", action='store_true', default=False, help="Don't add ions to neutralize")
    parser.add_argument("-e", "--equilibration-steps", type=int, default=200, help="Number of equilibration steps")
    parser.add_argument("--protein-force-field", default='amber/ff14SB.xml', help="Protein force field")
    parser.add_argument("--ligand-force-field", default='gaff-2.11', help="Ligand force field")
    parser.add_argument("--water-force-field", default='amber/tip3p_standard.xml', help="Water force field")
    
    args = parser.parse_args()
    return args


def create_complex(args, mol_in, pdb_in, output_complex):
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


# --- create_system small change: pass molecules as list ---
def create_system(platform, modeller, system_generator, args, ligand_mol, temperature):
    # pass molecules as a list (matches how SystemGenerator was constructed)
    system = system_generator.create_system(modeller.topology, molecules=[ligand_mol])

    friction_coeff = args.friction_coeff / unit.picosecond
    step_size = args.step_size * unit.picoseconds
    integrator = LangevinIntegrator(temperature, friction_coeff, step_size)

    if args.solvate:
        system.addForce(openmm.MonteCarloBarostat(1 * unit.atmospheres, temperature, 25))

    simulation = Simulation(modeller.topology, system, integrator, platform=platform)
    return simulation


def openmm_run():
    args = input_argument()
    pdb_in, mol_in = args.protein, args.ligand
    output_base = args.output
    output_complex = output_base + '_complex.pdb'
    output_traj_dcd = output_base + '_traj.dcd'
    output_min = output_base + '_minimised.pdb'
    checkpoint_file = output_base + "_checkpoint.chk"

    num_steps = args.steps
    chunk_steps = args.chunk_steps
    step_size = args.step_size * unit.picoseconds
    temperature = args.temperature * unit.kelvin

    total_duration = (step_size * num_steps).value_in_unit(unit.nanoseconds)
    chunk_duration = (step_size * chunk_steps).value_in_unit(unit.nanoseconds)
    print(f"Total simulation steps: {num_steps} (~{total_duration:.3f} ns)")
    print(f"Chunk size: {chunk_steps} steps (~{chunk_duration:.3f} ns)")

    platform, ligand_mol, modeller, system_generator = create_complex(args, mol_in, pdb_in, output_complex)
    simulation = create_system(platform, modeller, system_generator, args, ligand_mol, temperature)

    # --- Persistent Reporters ---
    simulation.reporters.append(PDBReporter(output_base + '_traj.pdb', args.interval))
    simulation.reporters.append(DCDReporter(output_traj_dcd, args.interval, enforcePeriodicBox=True))
    simulation.reporters.append(StateDataReporter(
        "simulation_progress_mitragynine_rotate_2AZ5.csv",
        args.interval,
        step=True, potentialEnergy=True, temperature=True,
        separator=",",
        append=True  # append if file exists
    ))

    # Checkpointing
    if os.path.exists(checkpoint_file):
        print("Resuming from checkpoint...")
        with open(checkpoint_file, "rb") as f:
            simulation.context.loadCheckpoint(f.read())
    else:
        print("No checkpoint found. Starting minimization + equilibration...")
        simulation.context.setPositions(modeller.positions)
        simulation.minimizeEnergy()
        # save minimized structure
        with open(output_min, 'w') as outfile:
            PDBFile.writeFile(
                modeller.topology,
                simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(),
                outfile)
        simulation.context.setVelocitiesToTemperature(temperature)
        simulation.step(args.equilibration_steps)

    # --- Run in chunks ---
    steps_done = 0
    chunk = 0
    while steps_done < num_steps:
        run_steps = min(chunk_steps, num_steps - steps_done)
        run_duration = (step_size * run_steps).value_in_unit(unit.nanoseconds)
        chunk += 1
        print(f"Running chunk {chunk}: {run_steps} steps (~{run_duration:.3f} ns)")

        t_start = time.time()
        simulation.step(run_steps)
        t_end = time.time()

        # Save checkpoint after each chunk
        with open(checkpoint_file, "wb") as f:
            f.write(simulation.context.createCheckpoint())

        steps_done += run_steps
        total_ns = (step_size * steps_done).value_in_unit(unit.nanoseconds)
        chunk_minutes = (t_end - t_start) / 60

        print(f"Completed {steps_done}/{num_steps} steps (~{total_ns:.3f} ns total)")
        print(f"Chunk took {chunk_minutes:.2f} minutes")

    print(f"Simulation complete! Total duration: ~{total_duration:.3f} ns")

if __name__ == "__main__":
    openmm_run()
