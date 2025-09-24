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

# Step 2: Loop through each file
for file_path in file_list:
    # Load data
    df = pd.read_csv(file_path, index_col=0)
    df = df.astype(float)
    model_names = df.index.tolist()

    # Create mask to hide upper triangle
    mask = np.triu(np.ones_like(df, dtype=bool))

    # Color coding: p < 0.05 = significant difference, else = not significant
    color_matrix = np.where(df < 0.05, 0, 1)
    cmap = ListedColormap(["#06068E", "#DCDBE7"])  # dark blue = significant, light gray = not significant

    # Plot heatmap (without showing p-values)
    plt.figure(figsize=(16, 12))
    ax = sns.heatmap(color_matrix,
                     mask=mask,
                     annot=False,       # <--- removed p-value annotations
                     cmap=cmap,
                     cbar=False,
                     linewidths=1,
                     square=True,
                     xticklabels=model_names,
                     yticklabels=model_names)

    # Title from file name (part after "InFlam_full_dunn_")
    metric_name = os.path.splitext(os.path.basename(file_path))[0].replace("InFlam_full_dunn_", "")
    plt.title(f"Dunn's Test Pairwise Comparison - {metric_name}", fontsize=12, weight='bold')

    plt.xticks(rotation=45, ha='right', fontsize=11, family='sans-serif')
    plt.yticks(rotation=0, fontsize=11, family='sans-serif')

    # Highlight the "XGB_RDKIT" model
    target_model = "XGB_RDKIT"
    if target_model in model_names:
        idx = model_names.index(target_model)
        rect_row = patches.Rectangle((0, idx), len(model_names), 1,
                                     fill=False,
                                     edgecolor='red',
                                     linewidth=1.8,
                                     linestyle='--')
        ax.add_patch(rect_row)

    # Add legend
    legend_elements = [
        patches.Patch(facecolor="#06068E", edgecolor="black", label="p < 0.05 (Significant difference)"),
        patches.Patch(facecolor="#DCDBE7", edgecolor="black", label="p ≥ 0.05 (Not significant)")
    ]
    ax.legend(handles=legend_elements,
              loc='upper right',
              bbox_to_anchor=(1.35, 1),  # place legend outside the plot
              frameon=True,
              fontsize=10)

    # Save SVG file
    output_file = os.path.join(input_dir, f"InFlam_full_dunn_{metric_name}_heatmap.svg")
    plt.tight_layout()
    plt.savefig(output_file, format="svg", dpi=300, bbox_inches="tight")
    plt.close()

print("✅ All heatmaps generated and saved with legends.")
