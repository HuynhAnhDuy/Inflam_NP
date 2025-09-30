import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# ====== Cấu hình ======
csv_path = "/home/andy/andy/Inflam_NP/5-CV/bilstm_pred_oof_5CV_maccs.csv"
output_folder = "confusion_5CV"
threshold = 0.5
labels = ['Negative', 'Positive']
# =======================

os.makedirs(output_folder, exist_ok=True)

# Đọc dữ liệu
df = pd.read_csv(csv_path)

# Tính accuracy cho toàn bộ dữ liệu
y_true = df['y_true'].astype(int)
y_pred = (df['y_predicted'] >= threshold).astype(int)
acc = accuracy_score(y_true, y_pred)
print(f"✅ Accuracy toàn bộ dữ liệu: {acc:.4f}")

# Hàm làm sạch tên file an toàn
def safe_name(s: str) -> str:
    s = re.sub(r"[^\w\-_. ]", "_", s)
    return s.strip().replace(" ", "_")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

# ==== % THEO TỪNG HÀNG (row-wise) ====
row_sums = cm.sum(axis=1, keepdims=True)
cm_percent = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100.0
# =====================================

# Tạo annotation: số đếm + %
annot = np.empty_like(cm).astype(str)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        annot[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"

plt.figure(figsize=(5, 5))
ax = sns.heatmap(
    cm,
    annot=annot,
    fmt="",
    cmap="Greys",
    cbar=False,
    xticklabels=labels,
    yticklabels=labels,
    linewidths=0.5,
    linecolor="black",
    annot_kws={"fontsize": 11, "weight": "bold"}
)

# Đảo màu chữ tùy theo nền
vmax = cm.max()
threshold_val = vmax / 2
for text, val in zip(ax.texts, cm.flatten()):
    text.set_color("white" if val > threshold_val else "black")

plt.xlabel('Predicted', fontsize=12, fontweight='bold', fontstyle='italic', family='sans-serif')
plt.ylabel('Actual', fontsize=12, fontweight='bold', fontstyle='italic', family='sans-serif')
plt.title(f'BiLSTM_MACCS with 5-CV', fontsize=12, fontweight='bold')
plt.tight_layout()

out_path = os.path.join(output_folder, f"confusion_matrix_BiLSTM_MACCS_5CV.svg")
plt.savefig(out_path, format="svg")
plt.close()
print(f"✅ Đã lưu: {out_path}")
