import pandas as pd

# ====== Đường dẫn các tập cuối ======
f1 = "1.Inflampred_modified.csv"
f2 = "2.AISMPred_modified.csv"
f3 = "3.InFlamNat_modified.csv"

key_col = "canonical_smiles"

# ====== Đọc dữ liệu ======
df1 = pd.read_csv(f1)
df2 = pd.read_csv(f2)
df3 = pd.read_csv(f3)

def internal_dups(df, name):
    """Trả về DataFrame các dòng trùng nội bộ theo key_col."""
    if key_col not in df.columns:
        raise ValueError(f"[{name}] không có cột {key_col}")
    # đánh dấu trùng (giữ toàn bộ bản ghi của những giá trị trùng)
    dups = df[df.duplicated(key_col, keep=False) & df[key_col].notna()] \
            .sort_values(key_col)
    return dups

# ====== 1) Trùng nội bộ trong từng tập ======
dups_1 = internal_dups(df1, "Inflampred")
dups_2 = internal_dups(df2, "AISMPred_final")
dups_3 = internal_dups(df3, "InFlamNat_final")

print("=== TRÙNG NỘI BỘ ===")
print("Inflampred: ", len(dups_1))
print("AISMPred_final: ", len(dups_2))
print("InFlamNat_final: ", len(dups_3))

# Lưu nếu có trùng nội bộ
if len(dups_1) > 0:
    dups_1.to_csv("dup_internal_inflampred.csv", index=False)
if len(dups_2) > 0:
    dups_2.to_csv("dup_internal_aismpred_final.csv", index=False)
if len(dups_3) > 0:
    dups_3.to_csv("dup_internal_inflamnat_final.csv", index=False)

# ====== 2) Trùng chéo giữa các tập ======
set1 = set(df1[key_col].dropna().unique())
set2 = set(df2[key_col].dropna().unique())
set3 = set(df3[key_col].dropna().unique())

overlap_12 = set1 & set2
overlap_13 = set1 & set3
overlap_23 = set2 & set3
overlap_123 = set1 & set2 & set3  # giao cả 3

print("\n=== TRÙNG CHÉO GIỮA CÁC TẬP ===")
print("Inflampred ↔ AISMPred_final:", len(overlap_12))
print("Inflampred ↔ InFlamNat_final:", len(overlap_13))
print("AISMPred_final ↔ InFlamNat_final:", len(overlap_23))
print("Giao của cả 3 tập:", len(overlap_123))

# Xuất danh sách trùng chéo (nếu có)
def save_overlap(smiles_set, out_name):
    if len(smiles_set) > 0:
        pd.DataFrame(sorted(smiles_set), columns=[key_col]).to_csv(out_name, index=False)

save_overlap(overlap_12, "overlap_inflampred__aismpred_final.csv")
save_overlap(overlap_13, "overlap_inflampred__inflamnat_final.csv")
save_overlap(overlap_23, "overlap_aismpred_final__inflamnat_final.csv")
save_overlap(overlap_123, "overlap_all_three.csv")

# ====== 3) Tóm tắt gọn ======
summary_rows = [
    ["dup_internal_inflampred", len(dups_1)],
    ["dup_internal_aismpred_final", len(dups_2)],
    ["dup_internal_inflamnat_final", len(dups_3)],
    ["overlap_inflampred__aismpred_final", len(overlap_12)],
    ["overlap_inflampred__inflamnat_final", len(overlap_13)],
    ["overlap_aismpred_final__inflamnat_final", len(overlap_23)],
    ["overlap_all_three", len(overlap_123)],
]
summary_df = pd.DataFrame(summary_rows, columns=["check", "count"])
summary_df.to_csv("final_overlap_report.csv", index=False)

print("\n=== TÓM TẮT ===")
print(summary_df)
if summary_df["count"].sum() == 0:
    print("\n✅ Không phát hiện bất kỳ trùng lặp nào (nội bộ hoặc chéo).")
else:
    print("\n⚠️ Có trùng lặp. Xem các file CSV *overlap_* và *dup_internal_* cùng 'final_overlap_report.csv' để chi tiết.")
