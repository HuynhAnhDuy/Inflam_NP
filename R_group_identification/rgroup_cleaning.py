import pandas as pd
import re

# === 1. Đọc file gốc ===
df = pd.read_csv("rgroup_only_enriched.csv")

# === 2. Hàm bỏ attachment [*:n] ===
def strip_attachment(smi: str) -> str:
    if pd.isna(smi):
        return smi
    return re.sub(r"\[\*:[0-9]+\]", "", smi)

# === 3. Tạo thêm cột mới "clean_smi" ===
df["clean_smi"] = df["rgroup_smiles"].apply(strip_attachment)

# === 4. Xuất file mới ===
df.to_csv("rgroup_only_enriched_clean.csv", index=False)

print("✅ Đã tạo file your_output.csv với thêm cột 'clean_smi'")
