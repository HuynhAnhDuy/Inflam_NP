import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob
import os

# Step 1: Get list of CSV files
input_dir = "/home/andy/andy/Inflam_NP/Statistics/"
file_list = glob.glob(os.path.join(input_dir, "InFlam_full_dunn_*.csv"))

# Target model
target_model = "XGB_RDKIT"

# Define colors (consistent for bars + legend)
colors = {
    "sig": "#FF7F0E",   # p < 0.05
    "nsig": "#1F77B4"   # p >= 0.05
}

# Step 2: Loop through each file
for file_path in file_list:
    # Load data
    df = pd.read_csv(file_path, index_col=0)
    df = df.astype(float)

    if target_model not in df.index:
        print(f"⚠️ {target_model} not found in {file_path}, skipped.")
        continue

    # Extract p-values vs target model (row + column just in case)
    p_values = df.loc[target_model].copy()
    if target_model in df.columns:
        p_values = p_values.combine(df[target_model], min)

    # Remove self-comparison
    p_values = p_values.drop(target_model)

    # Sort values
    p_values = p_values.sort_values()

    # Assign colors based on significance
    bar_colors = [colors["sig"] if p < 0.05 else colors["nsig"] for p in p_values.values]

    # Plot bar chart
    plt.figure(figsize=(6, 6))
    bars = plt.bar(p_values.index, p_values.values, color=bar_colors)

    # Horizontal threshold line
    plt.axhline(0.05, color="black", linestyle="--", linewidth=1.2)

    # Add text label next to Y axis
    plt.text(2, 0.08, "p = 0.05",
         color="black", fontsize=11, fontweight='bold', family='sans-serif', 
         ha="right", va="center")


    # Legend (linked to the same color scheme)
    legend_elements = [
        patches.Patch(facecolor=colors["sig"], edgecolor="black", label="p < 0.05 (Significant difference)"),
        patches.Patch(facecolor=colors["nsig"], edgecolor="black", label="p ≥ 0.05 (Not significant)")
    ]
    plt.legend(handles=legend_elements, loc="upper left", fontsize=10, frameon=True)

    # Title
    metric_name = os.path.splitext(os.path.basename(file_path))[0].replace("InFlam_full_dunn_", "")
    plt.title(f"Comparison of {target_model} vs other models – {metric_name}",
          fontsize=14, weight="bold")
    # Axes labels
    plt.ylabel("Dunn's test p-value", fontsize=12, fontweight='bold', fontstyle='italic', family='sans-serif')
    plt.xlabel("Model", fontsize=12, fontweight='bold', fontstyle='italic', family='sans-serif')
    plt.xticks(rotation=45, ha="right", fontsize=10)

    # Save output
    output_file = os.path.join(input_dir, f"InFlam_full_dunn_{metric_name}_{target_model}.svg")
    plt.tight_layout()
    plt.savefig(output_file, format="svg", dpi=300, bbox_inches="tight")
    plt.close()

print("✅ All bar plots generated and saved.")
