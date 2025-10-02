#!/usr/bin/env python3
"""
Run MD for a protein–ligand complex with optional solvation.
Option B: Input = complex PDB (protein+ligand pose) + ligand.xml
Phases:
  - Minimization
  - Equilibration (longer, default 200k steps ~ 400 ps)
  - Production (NPT or NVT)
Outputs:
  <base>_complex.pdb
  <base>_minimised.pdb
  <base>_traj.dcd
  <base>_energies.csv
  <base>_checkpoint.chk
  <base>_timing.csv
"""

import os, sys, time, argparse, csv
import openmm
import openmm as mm
from openmm import app, unit, LangevinIntegrator
from openmm.app import (PDBFile, Modeller, Simulation, 
                        DCDReporter, StateDataReporter, ForceField, CheckpointReporter)

# -----------------------
# Argument parsing
# -----------------------
def input_argument():
    p = argparse.ArgumentParser(description="MD simulation script (complex + ligand.xml)")
    # IO
    p.add_argument("-c", "--complex", default=os.path.join('mols', "Ligand_5_out_complex_COX2_5IKR.pdb"),
                   help="Complex PDB file (protein + ligand pose)")
    p.add_argument("-x", "--ligand-xml", default=os.path.join('mols', "ligand.xml"),
                   help="Ligand XML parameter file")
    p.add_argument("-o", "--output",  default=os.path.join('outputs', "run_complex"),
                   help="Base name for outputs (no extension)")

    # MD core
    p.add_argument("-z", "--step-size", type=float, default=0.002,
                   help="Integrator step size (ps). 0.002 ps = 2 fs")
    p.add_argument("-f", "--friction-coeff", type=float, default=1.0,
                   help="Langevin friction (ps^-1)")
    p.add_argument("-t", "--temperature", type=float, default=300.0,
                   help="Target temperature (K)")
    p.add_argument("--seed", type=int, default=2025, help="Random seed")

    # Production control
    p.add_argument("-s", "--steps", type=int, default=50000,
                   help="Number of production steps (ignored if --duration-ns set)")
    p.add_argument("--duration-ns", type=float, default=None,
                   help="Target simulation duration in ns (overrides --steps)")
    p.add_argument("-i", "--interval", type=int, default=100,
                   help="Reporting interval (steps) for trajectory/state")

    # Equilibration
    p.add_argument("--equil-steps", type=int, default=200000,
                   help="Number of equilibration steps before production (default 200k ~ 400 ps)")

    # Solvation
    p.add_argument("--solvate", action='store_true', default=False, help="Add solvent box")
    p.add_argument("--padding", type=float, default=10.0, help="Padding (Å) for solvent box")
    p.add_argument("--water-model", default="tip3p",
                   choices=["tip3p", "spce", "tip4pew", "tip5p", "swm4ndp"],
                   help="Water model")
    p.add_argument("--positive-ion", default="Na+", help="Positive ion")
    p.add_argument("--negative-ion", default="Cl-", help="Negative ion")
    p.add_argument("--ionic-strength", type=float, default=0.15, help="Salt (M)")
    p.add_argument("--no-neutralize", action='store_true', default=False,
                   help="Don't neutralize system")

    # Force fields
    p.add_argument("--protein-force-field", default='amber/ff14SB.xml', help="Protein FF")
    p.add_argument("--water-force-field",  default='amber/tip3p_standard.xml', help="Water FF")

    # Platform
    p.add_argument("--platform", default="CPU", choices=["CPU", "CUDA", "OpenCL"],
                   help="Computation platform")

    return p.parse_args()

# -----------------------
# Platform selection
# -----------------------
def get_platform(preferred: str):
    try:
        plat = mm.Platform.getPlatformByName(preferred)
        print(f"[Info] Using platform: {preferred}")
        return plat
    except Exception:
        print("[Warn] Requested platform not found, falling back to CPU.")
        return mm.Platform.getPlatformByName("CPU")

# -----------------------
# Build complex
# -----------------------
def create_complex(args, output_complex):
    platform = get_platform(args.platform)

    print('Reading complex PDB...')
    complex_pdb = PDBFile(args.complex)

    modeller = Modeller(complex_pdb.topology, complex_pdb.positions)
    print('Atoms before solvation:', modeller.topology.getNumAtoms())

    forcefield = ForceField(args.protein_force_field,
                            args.water_force_field,
                            args.ligand_xml)

    if args.solvate:
        print('Adding solvent...')
        modeller.addSolvent(forcefield,
                            model=args.water_model,
                            padding=args.padding * unit.angstroms,
                            positiveIon=args.positive_ion,
                            negativeIon=args.negative_ion,
                            ionicStrength=args.ionic_strength * unit.molar,
                            neutralize=not args.no_neutralize)
        print('Atoms after solvation:', modeller.topology.getNumAtoms())

    with open(output_complex, 'w') as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f)

    return platform, modeller, forcefield

# -----------------------
# System + Simulation
# -----------------------
def create_system_and_simulation(platform, modeller, forcefield, args):
    system = forcefield.createSystem(modeller.topology,
                                     nonbondedMethod=app.PME,
                                     nonbondedCutoff=1*unit.nanometer,
                                     constraints=app.HBonds)

    step_size = args.step_size * unit.picoseconds
    friction = args.friction_coeff / unit.picosecond
    integrator = LangevinIntegrator(args.temperature * unit.kelvin, friction, step_size)
    integrator.setRandomNumberSeed(args.seed)

    if args.solvate:
        system.addForce(openmm.MonteCarloBarostat(1.0 * unit.atmosphere,
                                                  args.temperature * unit.kelvin, 25))

    simulation = Simulation(modeller.topology, system, integrator, platform=platform)
    simulation.context.setPositions(modeller.positions)

    if system.usesPeriodicBoundaryConditions():
        print('[Info] Periodic box vectors:', system.getDefaultPeriodicBoxVectors())
    else:
        print('[Info] No periodic box')

    return simulation, system

# -----------------------
# Run
# -----------------------
def run_md():
    args = input_argument()
    t0 = time.time()

    # Prepare outputs
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    base = args.output
    out_complex = base + "_complex.pdb"
    out_min = base + "_minimised.pdb"
    out_traj = base + "_traj.dcd"
    out_csv  = base + "_timing.csv"
    out_ener = base + "_energies.csv"
    out_chk  = base + "_checkpoint.chk"

    print('Processing complex:', args.complex)
    print('Ligand XML:', args.ligand_xml)
    print('Outputs:', out_complex, out_min, out_traj, out_csv, out_ener, out_chk)

    # Build complex
    platform, modeller, forcefield = create_complex(args, out_complex)

    # System + Simulation
    simulation, system = create_system_and_simulation(platform, modeller, forcefield, args)

    # Minimization
    print('--- Minimization ---')
    tmin0 = time.time()
    simulation.minimizeEnergy(maxIterations=500)  # giới hạn vòng lặp
    with open(out_min, 'w') as f:
        PDBFile.writeFile(simulation.topology,
                          simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(),
                          file=f, keepIds=True)
    tmin1 = time.time()

    # Equilibration
    print('--- Equilibration ---')
    simulation.context.setVelocitiesToTemperature(args.temperature * unit.kelvin, args.seed)
    tequil0 = time.time()
    simulation.step(args.equil_steps)
    tequil1 = time.time()

    # Determine production steps
    if args.duration_ns is not None:
        prod_steps = int((args.duration_ns * 1000.0) / args.step_size)
        print(f"[Plan] Production: {args.duration_ns} ns -> {prod_steps} steps (dt={args.step_size} ps)")
    else:
        prod_steps = args.steps
        print(f"[Plan] Production: {prod_steps} steps "
              f"(dt={args.step_size} ps, ~{prod_steps*args.step_size/1000.0:.3f} ns)")

    # Production
    print('--- Production ---')
    simulation.reporters.append(DCDReporter(out_traj, args.interval, enforcePeriodicBox=True))
    simulation.reporters.append(StateDataReporter(sys.stdout, args.interval*5,
                                                  step=True, potentialEnergy=True,
                                                  temperature=True, pressure=True,
                                                  density=True, progress=True, remainingTime=True,
                                                  speed=True, totalSteps=prod_steps))
    simulation.reporters.append(StateDataReporter(out_ener, args.interval,
                                                  step=True, potentialEnergy=True,
                                                  totalEnergy=True, temperature=True,
                                                  pressure=True, density=True, separator=","))
    simulation.reporters.append(CheckpointReporter(out_chk, args.interval*50))

    tprod0 = time.time()
    simulation.step(prod_steps)
    tprod1 = time.time()

    # Summary
    t1 = time.time()
    total_ns = (prod_steps * args.step_size) / 1000.0
    print('All done. Wall clock time: {} min'.format(round((t1 - t0) / 60, 3)))
    print('Total simulated time: {:.3f} ns'.format(total_ns))

    # Write CSV timing
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Phase", "WallTime_min"])
        writer.writerow(["Minimization", round((tmin1 - tmin0)/60, 3)])
        writer.writerow(["Equilibration", round((tequil1 - tequil0)/60, 3)])
        writer.writerow(["Production", round((tprod1 - tprod0)/60, 3)])
        writer.writerow(["Total", round((t1 - t0)/60, 3)])
        writer.writerow(["Simulated_ns", total_ns])

    return "Successfully done"

if __name__ == "__main__":
    run_md()
