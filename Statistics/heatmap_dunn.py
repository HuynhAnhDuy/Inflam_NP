import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches

# Bước 1: Đọc dữ liệu
file_path = "/home/andy/andy/Inflam_NP/Statistics/InFlam_full_dunn_Specificity.csv"
df = pd.read_csv(file_path, index_col=0)
df = df.astype(float)
model_names = df.index.tolist()

# Bước 2: Tạo mask để ẩn tam giác trên
mask = np.triu(np.ones_like(df, dtype=bool))

# Bước 3: Ma trận tô màu
color_matrix = np.where(df < 0.05, 0, 1)
cmap = ListedColormap(["#121213", "#DCDBE7"])

# Bước 4: Vẽ heatmap
plt.figure(figsize=(12, 12))
ax = sns.heatmap(color_matrix,
                 mask=mask,
                 annot=df.round(3),
                 fmt=".3f",
                 cmap=cmap,
                 cbar=False,
                 linewidths=1,
                 square=True,
                 xticklabels=model_names,
                 yticklabels=model_names)

# Bước 5: Tùy chỉnh giao diện
plt.title("Dunn's Test Pairwise Comparison - Specificity", fontsize=12, weight='bold')
plt.xticks(rotation=45, ha='right', fontsize=11, family='sans-serif') 
plt.yticks(rotation=0, fontsize=11, family='sans-serif') 

# Bước 6: Đóng khung mô hình "XGB_RDKIT"
target_model = "XGB_RDKIT"
if target_model in model_names:
    idx = model_names.index(target_model)

    # Vẽ viền cho hàng
    rect_row = patches.Rectangle((0, idx), len(model_names), 1, 
                                 fill=False, 
                                 edgecolor='red', 
                                 linewidth=1.8,
                                 linestyle='--')
    ax.add_patch(rect_row)

# Bước 7: Lưu file
plt.tight_layout()
plt.savefig("InFlam_full_dunn_Specificity_heatmap.svg", format="svg", dpi=300)
