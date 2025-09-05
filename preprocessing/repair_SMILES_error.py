import re
import unicodedata
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
# Tuỳ chọn nâng cao (chuẩn hoá muối/điện tích)
try:
    from rdkit.Chem.MolStandardize import rdMolStandardize
    _HAS_STD = True
except Exception:
    _HAS_STD = False

# ------------- CẤU HÌNH -------------
INPUT_CSV   = "preprocessing/coconut_csv-09-2025.csv"          # Đổi thành file của bạn
SMILES_COL  = "SMILES"             # Đổi thành tên cột SMILES của bạn
OUTPUT_OK   = "coconut_clean.csv"      # (tuỳ chọn) file chỉ gồm dòng hợp lệ
OUTPUT_BAD  = "coconut_smiles_bad.csv"     # (tuỳ chọn) file ghi lại dòng hỏng

# Tắt bớt log RDKit
RDLogger.DisableLog('rdApp.error')

# Whitelist ký tự thường gặp trong SMILES (bạn có thể mở rộng nếu cần)
SMILES_CHARS = r"A-Za-z0-9@+\-\[\]\(\)=#\/\\.%:\*\."
RE_ONLY_SMILES = re.compile(fr"^[{SMILES_CHARS}]+$")
RE_DROP_NON_SMILES = re.compile(fr"[^{SMILES_CHARS}\s]")

def read_csv_safely(path: str) -> pd.DataFrame:
    """
    Đọc CSV sao cho giá trị không bị convert thành NaN/float bừa bãi.
    - dtype=str: giữ nguyên dạng chuỗi
    - keep_default_na=False: chuỗi 'NA', 'NaN' không tự thành NaN
    Thử utf-8, nếu lỗi thì fallback latin-1 (hoặc cp1252 tuỳ dữ liệu).
    """
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="latin-1")

def clean_smiles_text(s: str) -> str | None:
    """
    - Strip, chuẩn hoá Unicode (NFKC) để tránh ký tự look-alike.
    - Xoá ký tự không thuộc whitelist (kể cả emoji/điều khiển).
    - Xoá mọi khoảng trắng nội bộ (SMILES chuẩn không có space).
    """
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    s = unicodedata.normalize("NFKC", s)
    s = RE_DROP_NON_SMILES.sub("", s)
    s = re.sub(r"\s+", "", s)
    return s or None

def parse_mol_safe(smiles: str):
    """
    Parse an toàn: trả về None nếu lỗi. Có thể dùng sanitize=False để "cứu" thêm.
    """
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)  # hoặc Chem.MolFromSmiles(smiles, sanitize=False)
        # nếu dùng sanitize=False, có thể thử Chem.SanitizeMol(mol) trong try/except
        return mol
    except Exception:
        return None

def standardize_mol_optional(mol):
    """
    (Tuỳ chọn) Chuẩn hoá hoá học: bỏ muối/phân mảnh, uncharge → SMILES ổn định hơn.
    Cần RDKit >= 2020 và mô-đun MolStandardize.
    """
    if not _HAS_STD or mol is None:
        return mol
    try:
        # Bỏ phân mảnh nhỏ, chọn fragment chính
        lfc = rdMolStandardize.LargestFragmentChooser()
        mol = lfc.choose(mol)
        # Uncharge
        uc = rdMolStandardize.Uncharger()
        mol = uc.uncharge(mol)
        # Normalize (chuẩn hoá nhóm chức)
        normalizer = rdMolStandardize.Normalizer()
        mol = normalizer.normalize(mol)
        # Reionize
        reionizer = rdMolStandardize.Reionizer()
        mol = reionizer.reionize(mol)
        return mol
    except Exception:
        return mol

def canonicalize_smiles(mol):
    """
    Trả về canonical SMILES (giữ thông tin lập thể).
    """
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except Exception:
        return None

# ================== PIPELINE ==================
df = read_csv_safely(INPUT_CSV)

# Đảm bảo cột tồn tại
if SMILES_COL not in df.columns:
    raise ValueError(f"Không tìm thấy cột '{SMILES_COL}' trong CSV.")

# Làm sạch text
df["_smiles_clean"] = df[SMILES_COL].apply(clean_smiles_text)

# Parse RDKit
df["_mol"] = df["_smiles_clean"].apply(parse_mol_safe)

# (Tuỳ chọn) chuẩn hoá hoá học
df["_mol_std"] = df["_mol"].apply(standardize_mol_optional)

# Canonical SMILES
df["canonical_smiles"] = df["_mol_std"].apply(canonicalize_smiles)

# Đánh dấu hợp lệ
df["smiles_valid"] = df["_mol_std"].notna() & df["canonical_smiles"].notna()

# Tách dữ liệu tốt/xấu
df_ok  = df[df["smiles_valid"]].copy()
df_bad = df[~df["smiles_valid"]].copy()

# (Khuyến nghị) bỏ cột tạm
for c in ["_mol", "_mol_std", "smiles_valid"]:
    if c in df_ok.columns:
        df_ok.drop(columns=[c], inplace=True, errors="ignore")
    if c in df_bad.columns:
        df_bad.drop(columns=[c], inplace=True, errors="ignore")

# (Tuỳ chọn) Ghi ra file để kiểm tra/tiếp tục pipeline
df_ok.to_csv(OUTPUT_OK, index=False)
df_bad.to_csv(OUTPUT_BAD, index=False)

# Thống kê nhanh
summary = {
    "tổng_bản_ghi": len(df),
    "hợp_lệ": len(df_ok),
    "không_hợp_lệ": len(df_bad),
    "tỉ_lệ_hỏng(%)": round(100.0 * len(df_bad) / max(1, len(df)), 2),
}
print(summary)
