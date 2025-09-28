#!/usr/bin/env python3
import os, subprocess, re, math, csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ==== Cấu hình ====
RECEPTOR = "/home/andy/andy/Inflam_NP/molecular_docking/Protein_clean/COX1_1EQH_clean.pdbqt"
LIG_DIR = Path("ligands")
OUT_DIR = Path("COX1_1EQH_out_2")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Hộp docking (Å)
CENTER = (47.467, 27.863, 193.272)
SIZE   = (22, 22, 22)

# Tham số Vina
EXHAUSTIVENESS = 16
NUM_MODES = 9
ENERGY_RANGE = 3

# Tên lệnh vina (có thể là 'vina' hoặc 'autodock_vina')
VINA_BIN = "vina"

# Số luồng chạy song song
N_THREADS = 4

RT = 0.593  # kcal/mol ở 298K

def calc_Ki(deltaG_kcal):
    """Tính Ki (M) từ ΔG (kcal/mol)."""
    try:
        return math.exp(deltaG_kcal / RT)
    except Exception:
        return None

def run_vina(lig_path: Path):
    lig_base = lig_path.stem
    out_pose = OUT_DIR / f"{lig_base}_out.pdbqt"
    out_log  = OUT_DIR / f"{lig_base}.log"

    cmd = [
        VINA_BIN,
        "--receptor", RECEPTOR,
        "--ligand", str(lig_path),
        "--center_x", str(CENTER[0]), "--center_y", str(CENTER[1]), "--center_z", str(CENTER[2]),
        "--size_x", str(SIZE[0]), "--size_y", str(SIZE[1]), "--size_z", str(SIZE[2]),
        "--exhaustiveness", str(EXHAUSTIVENESS),
        "--num_modes", str(NUM_MODES),
        "--energy_range", str(ENERGY_RANGE),
        "--out", str(out_pose),
    ]
    try:
        with open(out_log, "w") as logf:
            subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError:
        return lig_base, []

    poses = []
    if out_log.exists():
        with open(out_log, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = re.match(r"\s*(\d+)\s+([-\d\.]+)", line)
                if m:
                    rank = int(m.group(1))
                    score = float(m.group(2))
                    if rank <= 3:  # chỉ lấy top 3
                        Ki = calc_Ki(score)
                        poses.append((lig_base, rank, score, Ki))
    return lig_base, poses

def main():
    ligands = sorted(LIG_DIR.glob("*.pdbqt"))
    if not ligands:
        print("Không tìm thấy ligand .pdbqt trong thư mục 'ligands/'.")
        return

    all_results = []
    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        fut2lig = {ex.submit(run_vina, lig): lig for lig in ligands}
        for fut in as_completed(fut2lig):
            lig_base, poses = fut.result()
            if not poses:
                print(f"[FAIL] {lig_base}: docking failed.")
            else:
                for lig, rank, score, Ki in poses:
                    if Ki:
                        print(f"[OK] {lig}: pose {rank}, affinity = {score:.3f} kcal/mol, Ki ≈ {Ki:.3e} M")
                    else:
                        print(f"[OK] {lig}: pose {rank}, affinity = {score:.3f} kcal/mol")
                all_results.extend(poses)

    # Xuất CSV
    csv_path = OUT_DIR / "scores.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fw:
        writer = csv.writer(fw)
        writer.writerow(["ligand", "pose_rank", "affinity (kcal/mol)", "Ki_estimated (M)"])
        for lig, rank, score, Ki in sorted(all_results):
            writer.writerow([lig, rank, f"{score:.3f}", "" if Ki is None else f"{Ki:.3e}"])

    print(f"\nHoàn tất. Xem kết quả & log trong '{OUT_DIR}/'. Bảng điểm CSV: {csv_path}")

if __name__ == "__main__":
    main()
