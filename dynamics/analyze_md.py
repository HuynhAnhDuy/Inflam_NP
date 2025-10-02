#!/usr/bin/env python3
"""
Analyze MD trajectory (manual input version):
 - RMSD
 - RMSF
 - Radius of gyration (Rg)
 - Solvent Accessible Surface Area (SASA)

Inputs: 
    - trajectory (.dcd/.xtc)
    - reference topology (.pdb/.gro/.prmtop)
Outputs:
    - CSV files saved in chosen output directory
"""

import os
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.analysis.rms import RMSD, RMSF
from MDAnalysis.analysis.radiusgyration import RadiusGyration
from MDAnalysis.analysis.sasa import SASA

# ============================
# >>> MANUAL INPUTS HERE <<<
# ============================
REF_FILE  = "outputs/run_complex_complex.pdb"   # Topology (PDB/GRO/PRMTOP)
TRAJ_FILE = "outputs/run_complex_traj.dcd"     # Trajectory (DCD/XTC)
OUTDIR    = "analysis_results"                 # Output directory
# ============================


def main():
    # Create output dir
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR, exist_ok=True)

    print(f"[Info] Loading trajectory: {TRAJ_FILE}")
    u = mda.Universe(REF_FILE, TRAJ_FILE)

    protein = u.select_atoms("protein")
    all_atoms = u.atoms

    # ---------- RMSD ----------
    print("[Task] RMSD (Cα vs first frame)...")
    R = RMSD(u, select="protein and name CA", ref_frame=0)
    R.run()
    rmsd_df = pd.DataFrame(R.rmsd, columns=["Frame", "Time_ps", "RMSD_Ang"])
    rmsd_df.to_csv(os.path.join(OUTDIR, "RMSD.csv"), index=False)

    # ---------- RMSF ----------
    print("[Task] RMSF (per-residue)...")
    align.AlignTraj(u, u, select="protein and name CA", in_memory=True).run()
    rmsf = RMSF(protein.select_atoms("name CA")).run()
    rmsf_df = pd.DataFrame({
        "Residue": [res.resid for res in protein.residues],
        "RMSF_Ang": rmsf.rmsf
    })
    rmsf_df.to_csv(os.path.join(OUTDIR, "RMSF.csv"), index=False)

    # ---------- Radius of gyration ----------
    print("[Task] Radius of gyration (whole system)...")
    Rg = RadiusGyration(all_atoms).run()
    rg_df = pd.DataFrame({
        "Frame": np.arange(len(Rg.results.rg)),
        "Rg_Ang": Rg.results.rg
    })
    rg_df.to_csv(os.path.join(OUTDIR, "Rg.csv"), index=False)

    # ---------- SASA ----------
    print("[Task] SASA (whole system)...")
    sasa = SASA(all_atoms).run()
    sasa_df = pd.DataFrame({
        "Frame": np.arange(len(sasa.results.sasa)),
        "SASA_A2": sasa.results.sasa
    })
    sasa_df.to_csv(os.path.join(OUTDIR, "SASA.csv"), index=False)

    print(f"[Done] Analysis finished. CSVs saved in: {OUTDIR}")


if __name__ == "__main__":
    main()
