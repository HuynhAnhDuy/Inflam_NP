import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

# --- Configuration ---
FILE_X_TRAIN = "InFlam_full_x_train.csv"
FILE_X_TEST = "InFlam_full_x_test.csv"
OUTPUT_SVG = "similarity_histogram_annotated.svg"

# ==========================================
# Step 1: Load data and Generate Fingerprints
# ==========================================
print("Loading data...")
x_train = pd.read_csv(FILE_X_TRAIN)
x_test = pd.read_csv(FILE_X_TEST)

def smiles_to_fp(smiles_list, radius=2, n_bits=2048):
    fps = []
    valid_indices = []
    for idx, smi in enumerate(smiles_list):
        smi_str = str(smi)
        if smi_str and smi_str.lower() != 'nan':
            mol = Chem.MolFromSmiles(smi_str)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                fps.append(fp)
                valid_indices.append(idx)
    return fps, valid_indices

smiles_col = "canonical_smiles"
train_fps, _ = smiles_to_fp(x_train[smiles_col].values)
test_fps, test_valid_indices = smiles_to_fp(x_test[smiles_col].values)

# ==========================================
# Step 2: Compute Max Tanimoto Similarity
# ==========================================
print("Calculating Max Tanimoto similarity...")
max_tanimoto_scores = []
for t_fp in test_fps:
    similarities = DataStructs.BulkTanimotoSimilarity(t_fp, train_fps)
    max_sim = max(similarities) if similarities else 0.0
    max_tanimoto_scores.append(max_sim)

x_test_analyzed = x_test.iloc[test_valid_indices].copy()
x_test_analyzed["max_train_tanimoto"] = max_tanimoto_scores
total_valid_test = len(x_test_analyzed)

# ==========================================
# Step 3: Calculate counts for the 3 ranges
# ==========================================
# Range 1: <= 0.3
# Range 2: 0.3 < x < 0.7 (hoặc tùy biến ranh giới bạn muốn hiển thị)
# Range 3: >= 0.7
count_low = (x_test_analyzed["max_train_tanimoto"] <= 0.4).sum()
count_mid = ((x_test_analyzed["max_train_tanimoto"] > 0.4) & (x_test_analyzed["max_train_tanimoto"] < 0.7)).sum()
count_high = (x_test_analyzed["max_train_tanimoto"] >= 0.7).sum()

pct_low = (count_low / total_valid_test) * 100
pct_mid = (count_mid / total_valid_test) * 100
pct_high = (count_high / total_valid_test) * 100

print(f"\n--- Statistics ---")
print(f"Total: {total_valid_test}")
print(f"<= 0.4: {count_low} ({pct_low:.1f}%)")
print(f"0.4 - 0.7: {count_mid} ({pct_mid:.1f}%)")
print(f">= 0.7: {count_high} ({pct_high:.1f}%)")

# ==========================================
# Step 4: Plot Histogram with Annotations
# ==========================================
print(f"\nGenerating histogram and saving to {OUTPUT_SVG}...")
plt.figure(figsize=(9, 6), dpi=100)

# Vẽ histogram kết hợp KDE bằng seaborn
ax = sns.histplot(
    x_test_analyzed["max_train_tanimoto"],
    bins=30,
    kde=True,
    color="#4682B4",
    edgecolor="black",
    linewidth=0.7,
    line_kws={"color": "#1C39BB", "linewidth": 2}
)

# Thêm các đường phân cách tại mốc 0.3 và 0.7
plt.axvline(x=0.4, color="#4BA2E4", linestyle="--", linewidth=1.5, label="Threshold = 0.3")
plt.axvline(x=0.7, color="#EA4367", linestyle="--", linewidth=1.5, label="Threshold = 0.7")

# Ghi chú thích số lượng và % cho từng khoảng ngay trên biểu đồ
y_max = ax.get_ylim()[1]

plt.text(0.15, y_max * 0.75, f"<= 0.4\n{count_low} ({pct_low:.1f}%)", 
         ha='center', va='center', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor="#54585B", alpha=0.85))

plt.text(0.50, y_max * 0.75, f"0.4 - 0.7\n{count_mid} ({pct_mid:.1f}%)", 
         ha='center', va='center', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#54585B', alpha=0.85))

plt.text(0.85, y_max * 0.75, f">= 0.7\n{count_high} ({pct_high:.1f}%)", 
         ha='center', va='center', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#54585B', alpha=0.85))

# Tiêu đề và nhãn trục
plt.title("Distribution of test compounds by max Tanimoto similarity", fontsize=12, fontweight='bold', pad=15)
plt.xlabel("Max Tanimoto similarity", fontsize=12, fontweight='bold', fontstyle='italic')
plt.ylabel("Number of compounds", fontsize=12, fontweight='bold', fontstyle='italic')

plt.legend(loc="upper right")
plt.tight_layout()

# Lưu file ảnh
plt.savefig(OUTPUT_SVG, format='svg', dpi=300)
print("Done.")