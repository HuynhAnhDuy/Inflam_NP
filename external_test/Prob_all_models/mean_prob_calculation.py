import pandas as pd
from pathlib import Path

# =========================
# Config
# =========================
IN_DIR = Path(".")          # thư mục chứa các csv input
OUT_DIR = Path("outputs")   # thư mục xuất kết quả
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Nếu tên file bạn có dạng:
# InFlam_external_neg3times_test_prob_ecfp_run1.csv
# ..._run2.csv, ..._run3.csv
# thì giữ như dưới. (Nếu file bạn không có .csv trong tên, hãy thêm vào.)
PATTERNS = {
    "ecfp":  "InFlam_external_neg3times_test_prob_ecfp_run*.csv",
    "maccs": "InFlam_external_neg3times_test_prob_maccs_run*.csv",
    "rdkit": "InFlam_external_neg3times_test_prob_rdkit_run*.csv",
}

REQUIRED_COLS = ["y_true", "y_prob"]

# =========================
# Helpers
# =========================
def load_prob_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: thiếu cột {missing}. Có các cột: {list(df.columns)}")

    # Chuẩn hóa kiểu dữ liệu
    df = df[REQUIRED_COLS].copy()
    df["y_true"] = pd.to_numeric(df["y_true"], errors="raise")
    df["y_prob"] = pd.to_numeric(df["y_prob"], errors="raise")
    return df

def mean_ensemble_for_fingerprint(fingerprint: str, pattern: str) -> Path:
    files = sorted(IN_DIR.glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"Không tìm thấy file nào cho {fingerprint} theo pattern: {pattern}")
    if len(files) < 2:
        raise ValueError(f"{fingerprint}: cần >=2 file để tính trung bình, nhưng chỉ thấy {len(files)} file: {files}")

    dfs = [load_prob_csv(p) for p in files]

    # Check cùng số dòng
    n = len(dfs[0])
    for p, df in zip(files, dfs):
        if len(df) != n:
            raise ValueError(f"{fingerprint}: số dòng không khớp. {files[0].name}={n}, {p.name}={len(df)}")

    # Check y_true giống nhau tuyệt đối theo thứ tự hàng
    base_true = dfs[0]["y_true"].reset_index(drop=True)
    for p, df in zip(files[1:], dfs[1:]):
        cur_true = df["y_true"].reset_index(drop=True)
        if not base_true.equals(cur_true):
            # gợi ý chỗ sai (nếu có)
            diff_idx = (base_true != cur_true).to_numpy().nonzero()[0]
            example = diff_idx[0] if len(diff_idx) else None
            raise ValueError(
                f"{fingerprint}: y_true KHÔNG giống nhau giữa {files[0].name} và {p.name}. "
                f"Ví dụ khác tại dòng {example}: {base_true.iloc[example]} vs {cur_true.iloc[example]}"
            )

    # Stack y_prob và lấy mean theo hàng
    probs = pd.concat([df["y_prob"].reset_index(drop=True) for df in dfs], axis=1)
    mean_prob = probs.mean(axis=1)

    out = pd.DataFrame({
        "y_true": base_true,
        "y_prob": mean_prob
    })

    out_path = OUT_DIR / f"InFlam_external_neg3times_test_prob_{fingerprint}_mean_BiLSTM.csv"
    out.to_csv(out_path, index=False)
    print(f"[OK] {fingerprint}: dùng {len(files)} files -> {out_path}")
    return out_path

# =========================
# Run
# =========================
if __name__ == "__main__":
    outputs = {}
    for fp, pat in PATTERNS.items():
        outputs[fp] = mean_ensemble_for_fingerprint(fp, pat)

    print("\nDone. Output files:")
    for k, v in outputs.items():
        print(f" - {k}: {v}")
