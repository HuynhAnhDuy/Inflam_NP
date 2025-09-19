import pandas as pd
from rdkit import Chem
import re

SMILES_CANDIDATES = [
    "SMILES","smiles","iso_smiles","canonical_smiles",
    "CanSMILES","can_smiles","structure_smiles"
]

def pick_smiles_col(df):
    for c in df.columns:
        if c in SMILES_CANDIDATES or "smiles" in c.lower():
            return c
    return None

def is_cas(s):
    return bool(re.fullmatch(r"\d{2,7}-\d{2}-\d", str(s).strip()))

def parse_ok(s):
    m = Chem.MolFromSmiles(str(s))
    return m is not None

def extract_clean_smiles(x):
    """Làm sạch 1 ô có thể chứa SMILES hoặc list ngăn bằng |; bỏ nguồn/CAS."""
    if pd.isna(x): 
        return None
    s = str(x).strip()
    if is_cas(s):
        return None
    if "|" in s:
        for token in s.split("|"):
            token = token.strip()
            if not token or is_cas(token):
                continue
            if parse_ok(token):
                return token
        return None
    return s if parse_ok(s) else None

# --- Đọc file gốc ---
data = pd.read_csv("coconut_csv-08-2025.csv", low_memory=False)
smicol = pick_smiles_col(data)
if smicol is None:
    raise ValueError("Không tìm thấy cột SMILES trong file NPASS.")

# --- Tạo cột SMILES sạch ---
data["SMILES_clean"] = data[smicol].apply(extract_clean_smiles)

# --- Ghi đè cột SMILES gốc bằng SMILES_clean ---
data[smicol] = data["SMILES_clean"]
data = data.drop(columns=["SMILES_clean"])

# --- Thống kê ---
total = len(data)
valid = data[smicol].notna().sum()
print(f"SMILES SCREENING: tổng {total} - SMILES hợp lệ: {valid}")

# --- Xuất 2 file ---
# 1) Giữ toàn bộ dataset (SMILES lỗi để trống)
data.to_csv("coconut_clean.csv", index=False)

# 2) Chỉ giữ dòng SMILES hợp lệ
data_valid = data[data[smicol].notna()].copy()
data_valid.to_csv("coconut_valid.csv", index=False)

print("Đã lưu:")
print(" - NPASSv2.0_clean.csv (giữ tất cả dòng, SMILES lỗi để trống)")
print(" - NPASSv2.0_only_valid.csv (chỉ các dòng có SMILES hợp lệ)")
