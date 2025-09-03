import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

# ====== Cấu hình ======
csv_path = "/home/andy/andy/Inflam_NP/visualization/InFlam_full_test_all_probs.csv"
output_path = "InFlam_full_test_AUPRC_plot.svg"
colormap_name = 'tab10'  # 🔁 Bạn có thể thử: 'Dark2', 'tab10', 'Paired', ...
# =======================

# Đọc dữ liệu
df = pd.read_csv(csv_path)
models = df['model'].unique()

# Lấy colormap
cmap = plt.get_cmap(colormap_name)
colors = [cmap(i % cmap.N) for i in range(len(models))]

# Khởi tạo figure
plt.figure(figsize=(6, 6))

# Vẽ các đường PR curve
for idx, model in enumerate(models):
    data = df[df['model'] == model]
    y_true = data['y_true']
    y_score = data['y_prob']

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auprc = auc(recall, precision)

    plt.plot(recall, precision,
             label=f'{model} (AUPRC={auprc:.3f})',
             color=colors[idx],
             linewidth=1.8,
             linestyle='-')

# Tùy chỉnh biểu đồ
plt.xlabel('Recall', fontsize=12, fontweight='bold', fontstyle='italic', family='sans-serif') 
plt.ylabel('Precision', fontsize=12, fontweight='bold', fontstyle='italic', family='sans-serif') 
plt.title('Precision-Recall Curve (AUPRC) for 20 models', fontsize=12, fontweight='bold', family='sans-serif') 
plt.legend(loc='lower left', fontsize='small', ncol=2)
plt.grid(True)

# Lưu biểu đồ
plt.tight_layout()
plt.savefig(output_path, format='svg')
print(f"✅ Đã lưu file: {output_path}")
