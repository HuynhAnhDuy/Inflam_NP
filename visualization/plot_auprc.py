import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

# ====== Cấu hình ======
csv_path = "InFlam_full_test_all_probs.csv"
output_path = "InFlam_test_AUPRC_plot.svg"
colormap_name = 'tab10'  # 🔁 Bạn có thể thử: 'Dark2', 'tab10', 'Paired', ...
# =======================

# Đọc dữ liệu
df = pd.read_csv(csv_path)
models = df['model'].unique()

# Tính precision-recall và AUPRC cho từng model
pr_data = []
for model in models:
    data = df[df['model'] == model]
    y_true = data['y_true']
    y_score = data['y_prob']

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auprc = auc(recall, precision)

    pr_data.append((model, recall, precision, auprc))

# Sắp xếp theo AUPRC giảm dần
pr_data_sorted = sorted(pr_data, key=lambda x: x[3], reverse=True)

# Lấy colormap
cmap = plt.get_cmap(colormap_name)
colors = [cmap(i % cmap.N) for i in range(len(pr_data_sorted))]

# Khởi tạo figure
plt.figure(figsize=(8, 6))

# Vẽ từng đường PR theo thứ tự AUPRC
for idx, (model, recall, precision, auprc) in enumerate(pr_data_sorted):
    plt.plot(recall, precision,
             label=f'{model} (AUPRC={auprc:.3f})',
             color=colors[idx],
             linewidth=1.8,
             linestyle='-')

# Tùy chỉnh biểu đồ
plt.xlabel('Recall', fontsize=12, fontweight='bold', fontstyle='italic', family='sans-serif') 
plt.ylabel('Precision', fontsize=12, fontweight='bold', fontstyle='italic', family='sans-serif') 
plt.title('AUPRC for 20 predictive models', fontsize=12, fontweight='bold', family='sans-serif') 
plt.legend(loc='lower left', fontsize='10', ncol=2)
plt.grid(True)

# Lưu biểu đồ
plt.tight_layout()
plt.savefig(output_path, format='svg')
print(f"✅ Đã lưu file: {output_path}")
