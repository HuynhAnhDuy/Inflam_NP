import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

# Step 1: Load data from CSV files
x_train = pd.read_csv("InFlam_full_x_train.csv")
y_train = pd.read_csv("InFlam_full_y_train.csv")
x_test = pd.read_csv("InFlam_full_x_test.csv")
y_test = pd.read_csv("InFlam_full_y_test.csv")


# Helper function to convert SMILES into RDKit Morgan Fingerprint (ECFP4)
def smiles_to_fp(smiles_list, radius=2, n_bits=2048):
    fps = []
    valid_indices = []
    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            # Generate Morgan fingerprint with radius 2 (equivalent to ECFP4)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fps.append(fp)
            valid_indices.append(idx)
    return fps, valid_indices


print("Generating fingerprints for Train and Test sets...")
train_fps, _ = smiles_to_fp(x_train["canonical_smiles"].values)
test_fps, test_valid_indices = smiles_to_fp(x_test["canonical_smiles"].values)

# Step 2: Compute Residual Structural Similarity (Max Tanimoto Similarity)
print("Calculating Tanimoto similarity matrix...")
max_tanimoto_scores = []

for t_fp in test_fps:
    # Compute Tanimoto similarity between a test fingerprint and all train fingerprints
    similarities = DataStructs.BulkTanimotoSimilarity(t_fp, train_fps)
    max_sim = max(similarities) if similarities else 0.0
    max_tanimoto_scores.append(max_sim)

# Attach max_tanimoto results to a copy of valid x_test rows
x_test_analyzed = x_test.iloc[test_valid_indices].copy()
x_test_analyzed["max_train_tanimoto"] = max_tanimoto_scores

# Step 3: Filter out test compounds with Tanimoto > 0.7
THRESHOLD = 0.7
filtered_mask = x_test_analyzed["max_train_tanimoto"] <= THRESHOLD
valid_filtered_indices = x_test_analyzed[filtered_mask].index

x_test_modified = x_test.loc[valid_filtered_indices].reset_index(drop=True)
y_test_modified = y_test.loc[valid_filtered_indices].reset_index(drop=True)

# Calculate statistics for visualization and logging
total_test_samples = len(x_test_analyzed)
num_retained = filtered_mask.sum()
num_removed = total_test_samples - num_retained
pct_retained = (num_retained / total_test_samples) * 100
pct_removed = (num_removed / total_test_samples) * 100

print(f"Initial Test samples: {total_test_samples}")
print(
    f"Retained Test samples (<= {THRESHOLD}): {num_retained} ({pct_retained:.1f}%)"
)
print(f"Removed Test samples (> {THRESHOLD}): {num_removed} ({pct_removed:.1f}%)")

# Step 4: Plot distribution with annotated statistics
plt.figure(figsize=(8, 6))
sns.histplot(
    x_test_analyzed["max_train_tanimoto"],
    bins=30,
    kde=True,
    color="#4682B4",  # Steel Blue thanh lịch
    edgecolor="black",  # Viền trắng giữa các cột giúp biểu đồ sắc nét hơn
    linewidth=0.7,
    line_kws={"color": "#1C39BB", "linewidth": 2},  # Đường KDE màu xanh đậm nổi bật
)

plt.axvline(
    x=THRESHOLD,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Cutoff = {THRESHOLD}",
)

# Text box formatting for counts and percentages
text_str = (
    f"Total test molecules: {total_test_samples}\n"
    f"Retained (<= {THRESHOLD}): {num_retained} ({pct_retained:.1f}%)\n"
    f"Removed (> {THRESHOLD}): {num_removed} ({pct_removed:.1f}%)"
)

props = dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray")
plt.gca().text(
    0.05,
    0.95,
    text_str,
    transform=plt.gca().transAxes,
    fontsize=11,
    verticalalignment="top",
    bbox=props,
)
plt.xlabel("Max Tanimoto similarity", fontsize=12, fontweight="bold", fontstyle="italic")
plt.ylabel("Number of compounds", fontsize=12, fontweight="bold", fontstyle="italic")
plt.legend(loc="upper right")
plt.tight_layout()

# Save figure as SVG for high-quality publication and CSV files
plt.savefig("residual_structural_similarity.svg", format="svg", dpi=300)

x_test_modified.to_csv("InFlam_modified_x_test.csv", index=False)
y_test_modified.to_csv("InFlam_modified_y_test.csv", index=False)
print(
    "Saved filtered datasets to 'InFlam_x_test_modified.csv' and 'InFlam_y_test_modified.csv'."
)