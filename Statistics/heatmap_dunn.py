import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import glob
import os

# Step 1: Get list of CSV files
input_dir = "/home/andy/andy/Inflam_NP/Statistics/"
file_list = glob.glob(os.path.join(input_dir, "InFlam_full_dunn_*.csv"))

def normalize_model_name(name: str) -> str:
    s = str(name).strip()
    s = s.replace("RDKit", "RDKIT").replace("Rdkit", "RDKIT").replace("rdkit", "RDKIT")
    return s

def p_to_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""

for file_path in file_list:
    # Load data
    df = pd.read_csv(file_path, index_col=0)

    # normalize labels (optional but helps matching consistency)
    df.index = [normalize_model_name(i) for i in df.index]
    df.columns = [normalize_model_name(c) for c in df.columns]

    # numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # ensure square matrix if needed (common case is already square)
    # If your CSV is square and aligned, you can omit these two lines.
    common = [m for m in df.index if m in df.columns]
    df = df.loc[common, common].copy()

    model_names = df.index.tolist()

    # symmetrize to be safe (min of ij and ji)
    df = df.combine(df.T, func=lambda a, b: np.fmin(a, b))

    # remove diagonal (no self-comparison)
    np.fill_diagonal(df.values, np.nan)

    # Create mask to hide upper triangle (including diagonal)
    mask = np.triu(np.ones_like(df, dtype=bool), k=0)

    # Color coding: p < 0.05 = significant difference, else = not significant
    # (Treat NaN as not-significant color; diagonal is masked anyway)
    color_matrix = np.where((df.values < 0.05) & ~np.isnan(df.values), 0, 1)
    cmap = ListedColormap(["#BCBEBF", "#055384"])


    # Figure size auto-scale theo số model để đỡ rối
    n = len(model_names)
    fig_w = max(10, 0.45 * n)
    fig_h = max(8, 0.45 * n)

    plt.figure(figsize=(fig_w, fig_h))
    ax = sns.heatmap(
        color_matrix,
        mask=mask,
        annot=False,
        cmap=cmap,
        cbar=False,
        linewidths=0.8,
        linecolor="white",
        square=True,
        xticklabels=model_names,
        yticklabels=model_names
    )

    # Title from file name
    metric_name = os.path.splitext(os.path.basename(file_path))[0].replace("InFlam_full_dunn_", "")
    plt.title(f"Dunn's Test Pairwise Comparison - {metric_name}", fontsize=12, weight='bold')

    plt.xticks(rotation=45, ha='right', fontsize=10, family='sans-serif')
    plt.yticks(rotation=0, fontsize=10, family='sans-serif')

    # ---- Add stars + outline boxes for significant cells (lower triangle only) ----
    for i in range(n):
        for j in range(n):
            if j >= i:   # only lower triangle
                continue

            p = df.iat[i, j]
            if pd.isna(p):
                continue

            # stars
            stars = p_to_stars(p)
            if stars:
                ax.text(j + 0.5, i + 0.5, stars,
                        ha="center", va="center",
                        fontsize=12, fontweight="bold", color="black")

            # outline box for p < 0.05
            if p < 0.05:
                rect = patches.Rectangle(
                    (j, i), 1, 1,
                    fill=False,
                    edgecolor="white",
                    linewidth=1
                )
                ax.add_patch(rect)

    # Add legend
    legend_elements = [
        patches.Patch(facecolor="#BCBEBF", edgecolor="black", label="p < 0.05 (Significant difference)"),
        patches.Patch(facecolor="#055384", edgecolor="black", label="p ≥ 0.05 (Not significant)")
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        bbox_to_anchor=(1.35, 1),
        frameon=True,
        fontsize=10
    )

    # Save SVG file
    output_file = os.path.join(input_dir, f"InFlam_full_dunn_{metric_name}_heatmap.svg")
    plt.tight_layout()
    plt.savefig(output_file, format="svg", dpi=300, bbox_inches="tight")
    plt.close()

print("✅ All heatmaps generated (all models) with stars + outline boxes + legends.")
