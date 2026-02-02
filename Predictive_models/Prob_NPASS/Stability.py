import os
import re
import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt

# ====== CONFIG ======
FILES = [
    "/home/andy/andy/Inflam_NP/Predictive_models/Prob_NPASS/NPASS_test_pred_BiLSTM_ecfp.csv",
    "/home/andy/andy/Inflam_NP/Predictive_models/Prob_NPASS/NPASS_test_pred_BiLSTM_maccs.csv",
    "/home/andy/andy/Inflam_NP/Predictive_models/Prob_NPASS/NPASS_test_pred_BiLSTM_rdkit.csv",
    "/home/andy/andy/Inflam_NP/Predictive_models/Prob_NPASS/NPASS_test_pred_XGB_ecfp.csv",
    "/home/andy/andy/Inflam_NP/Predictive_models/Prob_NPASS/NPASS_test_pred_XGB_maccs.csv",
    "/home/andy/andy/Inflam_NP/Predictive_models/Prob_NPASS/NPASS_test_pred_XGB_rdkit.csv",
]
KEY = "canonical_smiles"

OUT_CSV = "jaccard_similarity_matrix_UPPERCASE.csv"
OUT_FIG = "jaccard_similarity_lower_triangle_heatmap_UPPERCASE.svg"
# ====================


def infer_model_name_upper(path: str) -> str:
    base = os.path.basename(path)
    base = re.sub(r"\.csv$", "", base)
    base = base.replace("NPASS_test_pred_", "")
    return base.upper()  # <-- IN HOA


# ====== 1) Load active sets for each model ======
model_sets = {}
model_names = []

for f in FILES:
    model = infer_model_name_upper(f)
    model_names.append(model)

    df = pd.read_csv(f)
    if KEY not in df.columns or "y_pred" not in df.columns:
        raise ValueError(f"File {f} must contain columns: {KEY}, y_pred")

    actives = set(df.loc[df["y_pred"] == 1, KEY].astype(str))
    model_sets[model] = actives

    print(f"{model}: {len(actives)} predicted actives")

# ====== 2) Compute Jaccard similarity matrix ======
n = len(model_names)
jaccard_mat = pd.DataFrame(
    np.zeros((n, n), dtype=float),
    index=model_names,
    columns=model_names
)

for m1, m2 in combinations(model_names, 2):
    A = model_sets[m1]
    B = model_sets[m2]
    inter = len(A & B)
    union = len(A | B)
    j = inter / union if union > 0 else 0.0

    jaccard_mat.loc[m1, m2] = j
    jaccard_mat.loc[m2, m1] = j

np.fill_diagonal(jaccard_mat.values, 1.0)

# save matrix
jaccard_mat.to_csv(OUT_CSV)
print("\nSaved Jaccard matrix:", OUT_CSV)
print(jaccard_mat.round(3))

# ====== 3) Plot lower-triangle heatmap (hide upper triangle + diagonal) ======
labels = jaccard_mat.index.tolist()
M = jaccard_mat.values

# mask upper triangle including diagonal -> show only lower triangle
mask = np.triu(np.ones_like(M, dtype=bool))
M_masked = M.copy()
M_masked[mask] = np.nan

fig = plt.figure()
ax = plt.gca()

im = ax.imshow(M_masked, vmin=0, vmax=1, cmap="magma")

ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_yticklabels(labels)

# annotate only visible cells (lower triangle)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if not mask[i, j]:
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center")

plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Jaccard similarity of predicted-active sets")
ax.set_xlabel("Predictive model",fontsize=12, fontweight='bold', fontstyle='italic', family='sans-serif')
ax.set_ylabel("Predictive model",fontsize=12, fontweight='bold', fontstyle='italic', family='sans-serif')

plt.tight_layout()
plt.savefig(OUT_FIG, format="svg")
plt.close(fig)

print("Saved lower-triangle heatmap:", OUT_FIG)
