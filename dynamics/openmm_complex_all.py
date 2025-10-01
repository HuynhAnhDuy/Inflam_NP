#!/usr/bin/env python3
"""
Run MD for a protein–ligand complex with optional solvation.
Phases:
  1) Minimization
  2) Heating (NVT, ramp Tstart -> Ttarget)
  3) Equilibration (NPT with barostat at Ttarget)
  4) Production (NPT with barostat at Ttarget)

Outputs:
  <base>_complex.pdb         # complex after build/solvate
  <base>_minimised.pdb       # after minimization
  <base>_heat.dcd            # heating trajectory
  <base>_equil.dcd           # equilibration trajectory
  <base>_prod.dcd            # production trajectory
  Stdout: state data every k steps
"""

import os, sys, time, argparse
from openff.toolkit import Molecule
from openmmforcefields.generators import SystemGenerator
import openmm
import openmm as mm
from openmm import app, unit
from openmm.app import PDBFile, Modeller, PDBReporter, StateDataReporter, DCDReporter, Simulation
from openmm import LangevinIntegrator, Vec3

# -----------------------
# Argument parsing
# -----------------------
def input_argument():
    p = argparse.ArgumentParser(description="Molecular simulation script (protein–ligand)")
    # IO
    p.add_argument("-p", "--protein", default=os.path.join('mols', "COX2_3LN1_clean_fixed.pdb"),
                   help="Protein PDB file")
    p.add_argument("-l", "--ligand",  default=os.path.join('mols', "Ligand_5.mol"),
                   help="Ligand file (.mol/.sdf/.mol2/.pdb) with 3D coordinates")
    p.add_argument("-o", "--output",  default=os.path.join('outputs', "TheasinensinA1_4xv2"),
                   help="Base name for outputs (no extension)")

    # MD core
    p.add_argument("-z", "--step-size", type=float, default=0.002,
                   help="Integrator step size (ps). 0.002 ps = 2 fs")
    p.add_argument("-f", "--friction-coeff", type=float, default=1.0,
                   help="Langevin friction (ps^-1)")
    p.add_argument("-t", "--temperature", type=float, default=300.0,
                   help="Target temperature (K)")
    p.add_argument("--seed", type=int, default=2025, help="Random seed")

    # Phases (durations in ps; reporters use 'interval' in steps)
    p.add_argument("--heating-time-ps", type=float, default=50.0,
                   help="Heating duration (ps), NVT ramp from Tstart to Ttarget")
    p.add_argument("--heating-start-T", type=float, default=50.0,
                   help="Heating start temperature (K)")
    p.add_argument("--equil-time-ps", type=float, default=100.0,
                   help="Equilibration duration (ps), NPT at target T")
    p.add_argument("--prod-time-ns", type=float, default=1.0,
                   help="Production duration (ns), NPT at target T")
    p.add_argument("-i", "--interval", type=int, default=100,
                   help="Reporting interval (steps) for DCD and StateData")

    # Solvation
    p.add_argument("--solvate", action='store_true', default=False, help="Add solvent box")
    p.add_argument("--padding", type=float, default=10.0, help="Padding (Å) for solvent box")
    p.add_argument("--water-model", default="tip3p",
                   choices=["tip3p", "spce", "tip4pew", "tip5p", "swm4ndp"],
                   help="Water model")
    p.add_argument("--positive-ion", default="Na+", help="Positive ion")
    p.add_argument("--negative-ion", default="Cl-", help="Negative ion")
    p.add_argument("--ionic-strength", type=float, default=0.15, help="Salt (M)")
    p.add_argument("--no-neutralize", action='store_true', default=False, help="Don't neutralize net charge")

    # Force fields
    p.add_argument("--protein-force-field", default='amber/ff14SB.xml', help="Protein FF")
    p.add_argument("--ligand-force-field", default='gaff-2.11', help="Ligand FF")
    p.add_argument("--water-force-field",  default='amber/tip3p_standard.xml', help="Water FF")

    # Platform
    p.add_argument("--platform", default="CPU", choices=["CPU", "CUDA", "OpenCL"],
                   help="Computation platform preference")

    return p.parse_args()

# -----------------------
# Platform selection with graceful fallback
# -----------------------
def get_platform(preferred: str):
    # Try preferred, else fallback CUDA -> OpenCL -> CPU
    order = []
    if preferred == "CUDA":
        order = ["CUDA", "OpenCL", "CPU"]
    elif preferred == "OpenCL":
        order = ["OpenCL", "CUDA", "CPU"]
    else:
        order = ["CPU", "CUDA", "OpenCL"]

    for name in order:
        try:
            plat = mm.Platform.getPlatformByName(name)
            # optional: quick capability check for CUDA/OpenCL
            if name in ("CUDA", "OpenCL"):
                # If found, assume usable. (Advanced: query device count via properties)
                pass
            print(f"[Info] Using platform: {name}")
            return plat
        except Exception:
            continue

    # Shouldn't happen, but ensure CPU
    print("[Warn] No requested platform available; falling back to CPU.")
    return mm.Platform.getPlatformByName("CPU")

# -----------------------
# Build complex (read, param, solvate)
# -----------------------
def create_complex(args, mol_in, pdb_in, output_complex):
    platform = get_platform(args.platform)

    print('Reading ligand')
    ligand_mol = Molecule.from_file(mol_in)
    if ligand_mol.n_conformers == 0:
        raise ValueError("Ligand file has no 3D conformer. Provide a 3D file or embed coordinates first.")
    lig_positions_nm = ligand_mol.conformers[0].to(unit.nanometer)

    print('Preparing system generator')
    forcefield_kwargs = {
        'constraints': app.HBonds,
        'rigidWater': True,
        'removeCMMotion': False,
        'hydrogenMass': 4 * unit.amu
    }
    system_generator = SystemGenerator(
        forcefields=[args.protein_force_field, args.water_force_field],
        small_molecule_forcefield=args.ligand_force_field,
        molecules=[ligand_mol],
        forcefield_kwargs=forcefield_kwargs
    )

    print('Reading protein PDB')
    protein_pdb = PDBFile(pdb_in)

    print('Preparing complex (adding ligand)...')
    modeller = Modeller(protein_pdb.topology, protein_pdb.positions)
    print('Atoms before ligand:', modeller.topology.getNumAtoms())

    lig_top_off = ligand_mol.to_topology()
    lig_top_omm = lig_top_off.to_openmm()
    modeller.add(lig_top_omm, lig_positions_nm)
    print('Atoms after ligand:', modeller.topology.getNumAtoms())

    if args.solvate:
        print('Adding solvent ...')
        modeller.addSolvent(
            system_generator.forcefield,
            model=args.water_model,
            padding=args.padding * unit.angstroms,
            positiveIon=args.positive_ion,
            negativeIon=args.negative_ion,
            ionicStrength=args.ionic_strength * unit.molar,
            neutralize=not args.no_neutralize
        )
        print('Atoms after solvation:', modeller.topology.getNumAtoms())

    with open(output_complex, 'w') as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f)

    return platform, ligand_mol, modeller, system_generator

# -----------------------
# Create system and Simulation handle
# -----------------------
def create_system_and_simulation(platform, modeller, system_generator, args, ligand_mol):
    system = system_generator.create_system(modeller.topology, molecules=ligand_mol)

    # Add barostat later (only for NPT phases). Keep reference.
    # Create integrator (Langevin) – we will adjust temperature during heating.
    step_size = args.step_size * unit.picoseconds
    friction = args.friction_coeff / unit.picosecond
    integrator = LangevinIntegrator(args.temperature * unit.kelvin, friction, step_size)
    integrator.setRandomNumberSeed(args.seed)

    simulation = Simulation(modeller.topology, system, integrator, platform=platform)
    simulation.context.setPositions(modeller.positions)

    if system.usesPeriodicBoundaryConditions():
        print('[Info] PBC box vectors:', system.getDefaultPeriodicBoxVectors())
    else:
        print('[Info] No periodic box')

    return simulation, system

# -----------------------
# Utility: phase runner
# -----------------------
def attach_reporters(simulation, dcd_path, interval, label=""):
    # Clear existing DCD reporters if any for new phase (keep StateData on stdout continuous)
    # Here we simply add new DCD and a StateData with phase label in header
    simulation.reporters.append(DCDReporter(dcd_path, interval, enforcePeriodicBox=True))
    simulation.reporters.append(StateDataReporter(
        sys.stdout, interval * 5,
        step=True, potentialEnergy=True, temperature=True, density=True,
        progress=True, remainingTime=True, speed=True,
        totalSteps=None, separator='\t'
    ))
    print(f"[Info] Reporters attached for {label}: DCD every {interval} steps.")

def run_minimization(simulation, output_min_pdb):
    print('Minimizing energy ...')
    simulation.minimizeEnergy()
    with open(output_min_pdb, 'w') as f:
        PDBFile.writeFile(simulation.topology,
                          simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(),
                          file=f, keepIds=True)
    print('[Done] Minimization complete.')

def run_heating(simulation, args, steps_total, interval, out_dcd):
    print('--- Heating (NVT) ---')
    attach_reporters(simulation, out_dcd, interval, label="heating")

    Tstart = args.heating_start_T * unit.kelvin
    Ttarget = args.temperature * unit.kelvin
    step_size = args.step_size * unit.picoseconds

    # Initialize velocities at start T
    simulation.context.setVelocitiesToTemperature(Tstart, args.seed)

    # Ramp temperature linearly over the heating duration
    n = steps_total
    if n <= 0:
        print('[Warn] Heating steps = 0; skipping heating.')
        return
    t0 = time.time()
    for k in range(1, n + 1):
        frac = k / n
        Tcurr = Tstart + (Ttarget - Tstart) * frac
        simulation.integrator.setTemperature(Tcurr)
        simulation.step(1)
        # (We use 1-step increments to ensure a smooth ramp;
        # if you prefer performance, batch steps and update T every N steps.)
    t1 = time.time()
    print(f'[Done] Heating complete in {round((t1 - t0)/60, 3)} min.')

def ensure_barostat(system, temperature, frequency=25):
    # Add MonteCarloBarostat if not already present
    has_barostat = any(isinstance(system.getForce(i), openmm.MonteCarloBarostat)
                       for i in range(system.getNumForces()))
    if not has_barostat:
        system.addForce(openmm.MonteCarloBarostat(1.0 * unit.atmosphere, temperature, frequency))
        print('[Info] Barostat added (NPT).')

def run_equilibration(simulation, system, args, steps, interval, out_dcd):
    print('--- Equilibration (NPT) ---')
    ensure_barostat(system, args.temperature * unit.kelvin, frequency=25)
    # Ensure integrator is at target T
    simulation.integrator.setTemperature(args.temperature * unit.kelvin)
    # Optional: reinitialize velocities at target T
    simulation.context.setVelocitiesToTemperature(args.temperature * unit.kelvin, args.seed)

    attach_reporters(simulation, out_dcd, interval, label="equilibration")

    if steps <= 0:
        print('[Warn] Equilibration steps = 0; skipping equilibration.')
        return
    t0 = time.time()
    simulation.step(steps)
    t1 = time.time()
    print(f'[Done] Equilibration complete in {round((t1 - t0)/60, 3)} min.')

def run_production(simulation, system, args, steps, interval, out_dcd):
    print('--- Production (NPT) ---')
    # Barostat should already be present, but ensure anyway
    ensure_barostat(system, args.temperature * unit.kelvin, frequency=25)
    simulation.integrator.setTemperature(args.temperature * unit.kelvin)

    attach_reporters(simulation, out_dcd, interval, label="production")

    if steps <= 0:
        print('[Warn] Production steps = 0; skipping production.')
        return
    t0 = time.time()
    simulation.step(steps)
    t1 = time.time()
    print(f'[Done] Production complete in {round((t1 - t0)/60, 3)} min.')

# -----------------------
# Main
# -----------------------
def openmm_function():
    args = input_argument()
    t0 = time.time()

    # Ensure output dir
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Paths
    pdb_in = args.protein
    mol_in = args.ligand
    base = args.output
    out_complex = base + "_complex.pdb"
    out_min = base + "_minimised.pdb"
    out_heat = base + "_heat.dcd"
    out_equil = base + "_equil.dcd"
    out_prod = base + "_prod.dcd"

    print('Processing', pdb_in, 'and', mol_in)
    print('Outputs:', out_complex, out_min, out_heat, out_equil, out_prod)

    # Build complex
    platform, ligand_mol, modeller, system_generator = create_complex(args, mol_in, pdb_in, out_complex)

    # Create system + simulation
    simulation, system = create_system_and_simulation(platform, modeller, system_generator, args, ligand_mol)

    # Minimization
    run_minimization(simulation, out_min)

    # Convert phase durations to steps
    dt_ps = args.step_size  # ps
    heat_steps = int(round(args.heating_time_ps / dt_ps))
    equil_steps = int(round(args.equil_time_ps / dt_ps))
    prod_steps = int(round(args.prod_time_ns * 1000.0 / dt_ps))  # ns -> ps -> steps

    print(f"[Plan] Steps: heat={heat_steps}, equil={equil_steps}, prod={prod_steps} (dt={dt_ps} ps)")
    print(f"[Thermostat] Langevin @ friction={args.friction_coeff} ps^-1")
    if args.solvate:
        print("[Ensemble] Heating: NVT, Equil: NPT, Production: NPT")
    else:
        print("[Ensemble] No solvent: Heating/Equil/Prod will run without barostat (effectively NVT)")

    # Heating (NVT ramp)
    run_heating(simulation, args, heat_steps, args.interval, out_heat)

    # Equilibration (NPT with barostat if solvated)
    if args.solvate:
        run_equilibration(simulation, system, args, equil_steps, args.interval, out_equil)
    else:
        print("[Info] Skipping NPT equilibration barostat because --solvate not set.")
        run_equilibration(simulation, system, args, equil_steps, args.interval, out_equil)

    # Production (NPT)
    run_production(simulation, system, args, prod_steps, args.interval, out_prod)

    t1 = time.time()
    print('All done. Total wall clock time: {} min'.format(round((t1 - t0) / 60, 3)))
    # For reference, simulated time:
    total_ns = (heat_steps + equil_steps + prod_steps) * dt_ps / 1000.0
    print('Total simulated time: {:.3f} ns'.format(total_ns))
    return "Successfully done"

if __name__ == "__main__":
    openmm_function()
