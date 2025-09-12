import pandas as pd
import re

# === 1. Đọc dữ liệu đầu vào ===
df = pd.read_csv("rgroup_enrichment_summary.csv")

# === 1b. Tiền xử lý: loại R-group “rác” ===
# Gom nhóm theo nguyên tố
atom_groups = {
    "H":  [r"^\[H\]\[\*\:\d+\]$"],
    "O":  [r"^\[O[+-]?\]\[\*\:\d+\]$", r"^O\[\*\:\d+\]$"],
    "N":  [r"^\[N[+-]?\]\[\*\:\d+\]$", r"^N\[\*\:\d+\]$"],
    "S":  [r"^\[S[+-]?\]\[\*\:\d+\]$", r"^S\[\*\:\d+\]$"],
    "C":  [r"^\[C[+-]?\]\[\*\:\d+\]$", r"^C\[\*\:\d+\]$"],
    "F":  [r"^F\[\*\:\d+\]$"],
    "Cl": [r"^\[Cl-?\]\[\*\:\d+\]$", r"^Cl\[\*\:\d+\]$"],
    "Br": [r"^\[Br-?\]\[\*\:\d+\]$", r"^Br\[\*\:\d+\]$"],
    "I":  [r"^\[I-?\]\[\*\:\d+\]$", r"^I\[\*\:\d+\]$"],
}

# Bổ sung pattern chung
general_patterns = [
    r"^\[\*\:\d+\]$",  # chỉ attachment point
    r"^[CONSFIBrcl]+\[\*\:\d+\](\.[CONSFIBrcl]+\[\*\:\d+\])+"
]

# Gom tất cả pattern
all_patterns = [pat for pats in atom_groups.values() for pat in pats] + general_patterns
bad_pattern_union = re.compile("|".join(all_patterns))

# Lọc bỏ R-group "rác"
df = df[~df['rgroup_smiles'].str.contains(bad_pattern_union, na=False)].copy()

# === 2. Hàm tiện ích lọc + tính score ===
def filter_and_score(data, condition, score_col="score"):
    out = data[condition].copy()
    out[score_col] = out['enrichment_score'] * out['active_count']
    return out.sort_values(score_col, ascending=False)

# (A) R-group chuyên biệt
df_specialized = filter_and_score(
    df,
    (df['active_count'] >= 5) & (df['inactive_count'] == 0)
)

# (B) R-group ưu thế
df_enriched = filter_and_score(
    df,
    (df['active_count'] > 0) & (df['inactive_count'] > 0) & (df['enrichment_score'] >= 2)
)

# === 3. Xuất CSV ===
df_specialized.to_csv("rgroup_specialized_only_active.csv", index=False)
df_enriched.to_csv("rgroup_enriched_mixed.csv", index=False)

# === 4. Thông báo kết quả ===
print("✅ Đã lưu:", "rgroup_specialized_only_active.csv")
print("✅ Đã lưu:", "rgroup_enriched_mixed.csv")
print(f"🔹 Số lượng R-group chuyên biệt: {len(df_specialized)}")
print(f"🔹 Số lượng R-group ưu thế: {len(df_enriched)}")
