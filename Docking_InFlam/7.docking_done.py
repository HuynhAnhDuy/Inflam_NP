#!/usr/bin/env python3
import os, subprocess, re, math, csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# =====================================================================
#                        1. KHỐI CẤU HÌNH
# =====================================================================

# --- ĐƯỜNG DẪN INPUT & OUTPUT ---
RECEPTOR = Path(r"D:\Andy\Inflam_NP\Docking_InFlam\Protein_prepared\mPGES1_5TL9_clean.pdbqt")
LIG_DIR = Path(r"D:\Andy\Inflam_NP\Docking_InFlam\ligands_mPGES1_5TL9")
REF_LIGAND_FILENAME = "7DN_ref.pdbqt" 
OUT_DIR = Path("Docking_results_mPGES1_5TL9_chainA2")
CSV_OUTPUT_NAME = "mPGES1_5TL9_comparison_scores_chainA2.csv"

# --- THÔNG SỐ HỘP GRID BOX (Å) ---
CENTER = (7.752, -17.305, 28.152)
SIZE   = (40, 40, 40)

# --- THAM SỐ CHẠY VINA ---
EXHAUSTIVENESS = 32
NUM_MODES = 10          # Số lượng pose xuất ra để đối chiếu
ENERGY_RANGE = 3
VINA_BIN = r"D:\Andy\mydocking\vina_1.2.7_win.exe"      
N_THREADS = 4          

# =====================================================================
#                     2. CÁC HÀM TÍNH TOÁN VÀ DOCKING
# =====================================================================

RT = 0.593  # kcal/mol ở 298K
OUT_DIR.mkdir(exist_ok=True, parents=True)

def calc_Ki(deltaG_kcal):
    try:
        return math.exp(deltaG_kcal / RT)
    except Exception:
        return None

def calculate_all_poses_rmsd_via_pymol(ref_path, pose_path):
    """Duyệt qua tất cả các state (pose) trong file kết quả và tính RMSD so với file gốc."""
    results_by_pose = {}
    try:
        import pymol
        pymol.pymol_argv = ['pymol', '-cq']
        pymol.finish_launching()
        from pymol import cmd
        
        cmd.delete("all")
        cmd.load(str(ref_path), "ref_lig")
        cmd.load(str(pose_path), "docked_poses")
        
        num_states = cmd.count_states("docked_poses")
        
        for i in range(1, num_states + 1):
            rmsd_val = cmd.align(f"docked_poses and state {i}", "ref_lig", cycles=0, transform=0)[0]
            results_by_pose[i] = rmsd_val
            
        cmd.delete("all")
    except Exception as e:
        print(f"[WARNING] Không thể tính RMSD tự động bằng PyMOL: {e}")
    return results_by_pose

def run_vina(lig_path: Path, is_reference: bool = False):
    lig_base = lig_path.stem
    suffix = "_REF" if is_reference else ""
    out_pose = OUT_DIR / f"{lig_base}{suffix}_out.pdbqt"
    out_log  = OUT_DIR / f"{lig_base}{suffix}.log"

    cmd = [
        VINA_BIN,
        "--receptor", str(RECEPTOR),
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
        return lig_base, None, [], is_reference

    poses = []
    if out_log.exists():
        with open(out_log, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = re.match(r"\s*(\d+)\s+([-\d\.]+)", line)
                if m:
                    rank = int(m.group(1))
                    score = float(m.group(2))
                    poses.append((rank, score))
                    
    best_pose = None
    if poses:
        best_rank, best_score = min(poses, key=lambda x: x[1])
        Ki = calc_Ki(best_score)
        best_pose = (lig_base, best_rank, best_score, Ki)
        
    return lig_base, best_pose, poses, is_reference

# =====================================================================
#                         3. LUỒNG CHẠY CHÍNH
# =====================================================================
def main():
    ref_ligand_path = LIG_DIR / REF_LIGAND_FILENAME
    all_files = sorted(LIG_DIR.glob("*.pdbqt"))
    
    test_ligands = []
    ref_exists = False

    for f in all_files:
        if f.name == REF_LIGAND_FILENAME:
            ref_exists = True
        else:
            test_ligands.append(f)

    if not ref_exists:
        print(f"[WARNING] Không tìm thấy file ligand gốc '{REF_LIGAND_FILENAME}' tại {LIG_DIR}")
        
    if not test_ligands and not ref_exists:
        print(f"[ERROR] Không tìm thấy bất kỳ file .pdbqt nào!")
        return

    tasks = []
    for lig in test_ligands:
        tasks.append((lig, False))
    if ref_exists:
        tasks.append((ref_ligand_path, True))

    print(f"[INFO] Bắt đầu chuẩn bị docking...")
    print("-" * 80)

    all_results = []
    
    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        fut2lig = {ex.submit(run_vina, path, is_ref): path for path, is_ref in tasks}
        for fut in as_completed(fut2lig):
            lig_base, best_pose, all_poses_score, is_ref = fut.result()
            role_str = "[Reference]" if is_ref else "[Test]"
            
            if not best_pose:
                print(f"[FAIL] {role_str} {lig_base}: Docking thất bại.")
            else:
                lig, rank, score, Ki = best_pose
                ki_str = f", Ki ≈ {Ki:.3e} M" if Ki else ""
                print(f"[OK] {role_str} {lig}: best pose {rank}, affinity = {score:.3f} kcal/mol{ki_str}")
                
                # ---- XỬ LÝ LƯU THÔNG TIN 10 POSE CHO CHẤT ĐỐI CHỨNG ----
                rmsd_summary = "N/A"
                if is_ref:
                    pose_file_path = OUT_DIR / f"{lig_base}_REF_out.pdbqt"
                    print(f"      ==> Đang tính RMSD tất cả các pose cho Reference...")
                    rmsd_dict = calculate_all_poses_rmsd_via_pymol(ref_ligand_path, pose_file_path)
                    score_dict = {p[0]: p[1] for p in all_poses_score}
                    
                    # Tạo chuỗi chi tiết gộp tất cả các pose lại để lưu vào CSV
                    pose_details = []
                    print(f"\n      {'POSE':<6} | {'Affinity (kcal/mol)':<20} | {'RMSD (Å)':<10}")
                    print("      " + "-" * 42)
                    for p_idx, sc in sorted(score_dict.items(), key=lambda x: x[0]):
                        r_val = rmsd_dict.get(p_idx, 0.0)
                        marker = " <--- ★ (< 2.0 Å)" if r_val <= 2.0 else ""
                        print(f"      Pose {p_idx:<2} | {sc:<20.3f} | {r_val:<10.3f}{marker}")
                        pose_details.append(f"Pose {p_idx}: {sc} kcal/mol, RMSD {r_val:.3f}A")
                    print("      " + "-" * 42 + "\n")
                    
                    rmsd_summary = " | ".join(pose_details)
                
                all_results.append((lig, rank, score, Ki, "Reference" if is_ref else "Test", rmsd_summary))

    # Xuất file CSV kết quả
    csv_path = OUT_DIR / CSV_OUTPUT_NAME
    with open(csv_path, "w", newline="", encoding="utf-8") as fw:
        writer = csv.writer(fw)
        writer.writerow(["ligand", "role", "best_pose_rank", "affinity (kcal/mol)", "Ki_estimated (M)", "All_Poses_Affinity_and_RMSD"])
        
        sorted_results = sorted(
            all_results, 
            key=lambda x: (0 if x[4] == "Reference" else 1, x[2])
        )
        
        for lig, rank, score, Ki, role, rmsd_info in sorted_results:
            writer.writerow([
                lig, 
                role, 
                rank, 
                f"{score:.3f}", 
                "" if Ki is None else f"{Ki:.3e}",
                rmsd_info
            ])

    print("-" * 80)
    print(f"[SUCCESS] Hoàn tất! File kết quả chi tiết kèm toàn bộ pose: {csv_path}")

if __name__ == "__main__":
    main()