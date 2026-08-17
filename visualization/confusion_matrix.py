import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# ====== Cấu hình ======
base_dir = r"D:\Andy\Inflam_NP\Predictive_models\Prob_InFlam_full"
output_folder = r"D:\Andy\Inflam_NP\visualization\confusion_6models"
threshold = 0.5
labels = ['Negative', 'Positive']

# Cấu hình cụ thể 6 mô hình cần vẽ (XGB và BiLSTM + RdKit, ECFP, MACCS)
target_models = {
    'XGB': [os.path.join(base_dir, "Prob_XGB")],
    'BiLSTM': [os.path.join(base_dir, "Prob_BiLSTM")]
}
features_to_include = ['rdkit', 'ecfp', 'maccs']
# =======================

os.makedirs(output_folder, exist_ok=True)

# Hàm làm sạch tên file an toàn
def safe_name(s: str) -> str:
    s = re.sub(r"[^\w\-_. ]", "_", s)
    return s.strip().replace(" ", "_")

# Thu thập dữ liệu từ các thư mục tương ứng
model_data_dict = {}

for model_name, folders in target_models.items():
    for folder_path in folders:
        if not os.path.exists(folder_path):
            print(f"⚠️ Cảnh báo: Không tìm thấy thư mục {folder_path}")
            continue
            
        for feat in features_to_include:
            # Tìm kiếm file qua 3 lần run (ví dụ: InFlam_full_test_prob_ecfp_run1.csv)
            search_pattern = os.path.join(folder_path, f"InFlam_full_test_prob_{feat}_run*.csv")
            run_files = glob.glob(search_pattern)
            
            if run_files:
                dfs = []
                for f in run_files:
                    temp_df = pd.read_csv(f)
                    dfs.append(temp_df)
                
                # Tính trung bình xác suất y_prob qua 3 lần run
                combined_df = dfs[0].copy()
                if 'y_prob' in combined_df.columns:
                    combined_df['y_prob'] = sum(d['y_prob'] for d in dfs) / len(dfs)
                
                display_name = f"{model_name} + {feat.upper()}"
                model_data_dict[display_name] = combined_df

# Vẽ và lưu từng hình độc lập cho 6 mô hình
for model_label, data in model_data_dict.items():
    y_true = data['y_true'].astype(int)
    y_pred = (data['y_prob'] >= threshold).astype(int)

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
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="black",
        annot_kws={"fontsize": 11, "weight": "bold"}
    )

    # Đảo màu chữ tùy theo độ đậm của ô
    vmax = cm.max()
    threshold_val = vmax / 2 if vmax > 0 else 0
    for text, val in zip(ax.texts, cm.flatten()):
        text.set_color("white" if val > threshold_val else "black")

    plt.xlabel('Predicted', fontsize=12, fontweight='bold', fontstyle='italic', family='sans-serif')
    plt.ylabel('Actual', fontsize=12, fontweight='bold', fontstyle='italic', family='sans-serif')
    plt.tight_layout()

    out_path = os.path.join(output_folder, f"confusion_matrix_{safe_name(model_label)}.svg")
    plt.savefig(out_path, format="svg", dpi=300)
    plt.close()
    print(f"✅ Đã lưu: {out_path}")