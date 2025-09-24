import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Bước 1: Load data
df = pd.read_csv('InFlam_full_all_metrics_heatmap.csv')
df.set_index('Model', inplace=True)

# Chỉ giữ 4 cột mong muốn
df = df[["Accuracy", "MCC", "Sensitivity", "Specificity"]]

# Bước 2: Thiết lập figure
plt.figure(figsize=(9, 6))

# Bước 3: Vẽ heatmap
ax = sns.heatmap(df,
                 cmap='RdYlBu',     # Colormap đẹp hơn: Vàng → Xanh
                 annot=True,        # Tự annotate luôn, khỏi viết loop tay
                 fmt=".3f",
                 linewidths=0.5,
                 linecolor="black",
                 cbar_kws={'label': 'Metric value'})

# Bước 4: Định dạng text (màu chữ tự động theo giá trị)
for text in ax.texts:
    val = float(text.get_text())
    text.set_color('black' if val < 0.91 else 'white')  # < 0.8 thì chữ đen, ngược lại chữ trắng
    text.set_fontsize(10)

# Bước 5: Nhãn trục
plt.yticks(rotation=0, fontsize=11, family='sans-serif')
plt.xticks(rotation=45, ha='right', fontsize=11, family='sans-serif')

# Bước 6: Tiêu đề (tuỳ chọn)
plt.title("Performance of predictive models", fontsize=12, fontweight='bold', family='sans-serif')

# Bước 7: Lưu và hiển thị
plt.tight_layout()
plt.savefig('InFlam_full_metrics_heatmap.svg', dpi=300, bbox_inches='tight')
