import pandas as pd
import re
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdRGroupDecomposition
import csv, gc

# === Config ===
min_compound_per_scaffold = 10
output_file = "rgroup_with_labels.csv"

# === Regex patterns loại bỏ R-group rác ===
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

general_patterns = [
    r"^\[\*\:\d+\]$",  # chỉ attachment point
    r"^[CONSFIBrcl]+\[\*\:\d+\](\.[CONSFIBrcl]+\[\*\:\d+\])+"
]

all_patterns = [pat for pats in atom_groups.values() for pat in pats] + general_patterns
bad_pattern_union = re.compile("|".join(all_patterns))

def is_garbage_rgroup(smi):
    return bool(bad_pattern_union.match(smi))

# === STEP 1: Đọc dữ liệu ===
print("📥 Đọc dữ liệu...")
df = pd.read_csv("InFlam_full.csv")  # cần có cột 'canonical_smiles' và 'Label'
df = df.dropna(subset=['canonical_smiles', 'Label'])
print(f"👉 Số dòng ban đầu: {len(df)}")

# Tạo molecule và scaffold
print("🔄 Chuyển SMILES thành molecule...")
df['mol'] = df['canonical_smiles'].apply(Chem.MolFromSmiles)
df = df[df['mol'].notnull()]
print(f"👉 Sau khi lọc mol hợp lệ: {len(df)}")

print("🔄 Sinh scaffold (MurckoScaffold)...")
df['scaffold'] = df['mol'].apply(lambda m: Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) if m else None)

# Lọc scaffold có ít nhất min_compound_per_scaffold
df_scaffold_counts = df['scaffold'].value_counts()
valid_scaffolds = df_scaffold_counts[df_scaffold_counts >= min_compound_per_scaffold].index
df = df[df['scaffold'].isin(valid_scaffolds)]

scaffolds = df['scaffold'].nunique()
print(f"👉 Sau khi lọc, còn {scaffolds} scaffold với ≥{min_compound_per_scaffold} compound")

# === STEP 2: Chuẩn bị file output (ghi từng phần) ===
processed = 0
skipped = 0

with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=['scaffold','rgroup_label','rgroup_smiles','compound_index','Label'])
    writer.writeheader()

    for idx, (scaffold_smiles, group) in enumerate(df.groupby('scaffold'), 1):
        if len(group) < min_compound_per_scaffold:
            skipped += 1
            print(f"⏩ Bỏ qua scaffold {idx}/{scaffolds} (chỉ có {len(group)} compound)")
            continue

        processed += 1
        print(f"🔹 Xử lý scaffold {idx}/{scaffolds} với {len(group)} compound...")

        try:
            scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
            if scaffold_mol is None:
                print("⚠️ Scaffold không hợp lệ, bỏ qua.")
                skipped += 1
                continue

            # === STEP 3: RGroup decomposition ===
            decomp = rdRGroupDecomposition.RGroupDecomposition([scaffold_mol])
            mols = group['mol'].tolist()
            for mol in mols:
                decomp.Add(mol)
            decomp.Process()
            rgroups_raw = decomp.GetRGroupsAsRows()

            # === STEP 4: Gắn với Label và ghi ra file ngay ===
            results_scaffold = []
            for i, rgroup in enumerate(rgroups_raw):
                label_val = group.iloc[i]['Label']
                for r_label, r_mol in rgroup.items():
                    if r_label == "Core":
                        continue
                    smi = Chem.MolToSmiles(r_mol) if r_mol else None
                    if smi and not is_garbage_rgroup(smi):  # lọc rác
                        results_scaffold.append({
                            'scaffold': scaffold_smiles,
                            'rgroup_label': r_label,
                            'rgroup_smiles': smi,
                            'compound_index': group.index[i],
                            'Label': label_val
                        })

            if results_scaffold:
                writer.writerows(results_scaffold)
                f.flush()

        except Exception as e:
            print(f"⚠️ Lỗi khi xử lý scaffold {scaffold_smiles}: {e}")
            skipped += 1
            continue

        # Giải phóng bộ nhớ
        del group, scaffold_smiles, scaffold_mol, mols, rgroups_raw, results_scaffold
        gc.collect()

print("📑 Hoàn tất trích xuất R-group, đã lưu từng phần vào:", output_file)

# === STEP 3: Tính enrichment sau khi đã có file R-group ===
print("📊 Tính enrichment...")
df_rgroup = pd.read_csv(output_file)

def compute_enrichment(df_rgroup):
    stats = []
    grouped = df_rgroup.groupby(['rgroup_label', 'rgroup_smiles'])
    total_1 = (df_rgroup['Label'] == 1).sum()
    total_0 = (df_rgroup['Label'] == 0).sum()

    for (r_label, r_smi), group in grouped:
        count_1 = (group['Label'] == 1).sum()
        count_0 = (group['Label'] == 0).sum()

        freq_1 = count_1 / total_1 if total_1 else 0
        freq_0 = count_0 / total_0 if total_0 else 0
        enrichment = (freq_1 / freq_0) if freq_0 > 0 else float('inf')

        stats.append({
            'rgroup_label': r_label,
            'rgroup_smiles': r_smi,
            'active_count': count_1,
            'inactive_count': count_0,
            'enrichment_score': enrichment
        })
    return pd.DataFrame(stats).sort_values('enrichment_score', ascending=False)

df_enrichment = compute_enrichment(df_rgroup)
df_enrichment.to_csv("rgroup_enrichment_summary_2.csv", index=False)

# === STEP 4: Gán nhãn enrichment (0/1 theo ngưỡng) ===
def assign_labels_enrichment(df, threshold_high=2, threshold_low=0.5):
    records = []
    for _, row in df.iterrows():
        if row['enrichment_score'] >= threshold_high:
            final_label = 1
        elif row['enrichment_score'] <= threshold_low:
            final_label = 0
        else:
            final_label = None  # trung tính
        records.append({**row, 'final_label_enrichment': final_label})
    return pd.DataFrame(records)

df_labeled_enrichment = assign_labels_enrichment(df_enrichment)
df_labeled_enrichment.to_csv("rgroup_labeled_enrichment.csv", index=False)

# === STEP 5: Gán nhãn majority vote ===
def assign_labels_majority(df_rgroup):
    grouped = df_rgroup.groupby(['rgroup_label','rgroup_smiles'])
    records = []
    for (r_label, smi), g in grouped:
        count_1 = (g['Label'] == 1).sum()
        count_0 = (g['Label'] == 0).sum()
        if count_1 > count_0:
            final_label = 1
        elif count_0 > count_1:
            final_label = 0
        else:
            final_label = None
        records.append({
            'rgroup_label': r_label,
            'rgroup_smiles': smi,
            'active_count': count_1,
            'inactive_count': count_0,
            'final_label_majority': final_label
        })
    return pd.DataFrame(records)

df_labeled_majority = assign_labels_majority(df_rgroup)
df_labeled_majority.to_csv("rgroup_labeled_majority.csv", index=False)

# === STEP 6: Summary cuối cùng ===
print("\n📊 Tóm tắt:")
print(f"- Scaffold được xử lý: {processed}")
print(f"- Scaffold bị bỏ qua (<{min_compound_per_scaffold} hoặc lỗi): {skipped}")
print(f"- Tổng scaffold đủ điều kiện: {scaffolds}")
print(f"- Tổng R-group trích xuất: {len(df_rgroup)}")
print(f"- Tổng R-group duy nhất: {df_enrichment.shape[0]}")

print("\n✅ Hoàn tất xử lý!")
print("- Đã lưu chi tiết R-group:", output_file)
print("- Đã lưu bảng enrichment: rgroup_enrichment_summary_2.csv")
print("- Đã lưu nhãn theo enrichment:", "rgroup_labeled_enrichment.csv")
print("- Đã lưu nhãn theo majority vote:", "rgroup_labeled_majority.csv")
