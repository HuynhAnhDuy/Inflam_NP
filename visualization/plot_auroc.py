import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ====== Cấu hình đường dẫn các thư mục mô hình ======
base_dir = r"D:\Andy\Inflam_NP\Predictive_models\Prob_InFlam_full"

model_dirs = {
    'RF': os.path.join(base_dir, "Prob_RF"),
    'XGB': os.path.join(base_dir, "Prob_XGB"),    
    'LGBM': os.path.join(base_dir, "Prob_LGBM"),  
    'BiLSTM': os.path.join(base_dir, "Prob_BiLSTM") 
}

features_to_include = ['ecfp', 'maccs', 'rdkit', 'estate', 'phychem']
output_path = "AUROC_plot.svg"
colormap_name = 'tab20'  # Dùng bảng màu 20 màu cho đủ các mô hình x features
# ====================================================

roc_data = []
plt.figure(figsize=(8, 8))

# Quét qua từng mô hình và từng loại feature
for model_name, folder_path in model_dirs.items():
    if not os.path.exists(folder_path):
        print(f"⚠️ Cảnh báo: Không tìm thấy thư mục {folder_path}")
        continue
        
    for feat in features_to_include:
        # Cập nhật pattern khớp chính xác với: InFlam_full_test_prob_{feat}_runX.csv
        search_pattern = os.path.join(folder_path, f"InFlam_full_test_prob_{feat}_run*.csv")
        run_files = glob.glob(search_pattern)
        
        if run_files:
            dfs = []
            for f in run_files:
                temp_df = pd.read_csv(f)
                dfs.append(temp_df)
            
            # Tính trung bình xác suất y_prob qua 3 lần run để lấy kết quả ổn định
            combined_df = dfs[0].copy()
            if 'y_prob' in combined_df.columns:
                combined_df['y_prob'] = sum(d['y_prob'] for d in dfs) / len(dfs)
            
            y_true = combined_df['y_true']
            y_score = combined_df['y_prob']
            
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auroc = auc(fpr, tpr)
            
            label_name = f"{model_name} + {feat.upper()}"
            roc_data.append((label_name, fpr, tpr, auroc))

# Sắp xếp theo AUROC giảm dần để thứ tự hiển thị trong Legend trực quan hơn
roc_data_sorted = sorted(roc_data, key=lambda x: x[3], reverse=True)

# Lấy colormap
cmap = plt.get_cmap(colormap_name)
colors = [cmap(i % cmap.N) for i in range(len(roc_data_sorted))]

# Vẽ từng đường ROC
for idx, (model_label, fpr, tpr, auroc) in enumerate(roc_data_sorted):
    plt.plot(fpr, tpr,
             label=f'{model_label} (AUROC={auroc:.3f})',
             color=colors[idx],
             linewidth=1.5,
             linestyle='-')

# Vẽ đường chéo ngẫu nhiên tham chiếu
plt.plot([0, 1], [0, 1], 'k--', linewidth=1.0, label='Random')

# Tùy chỉnh biểu đồ
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold', fontstyle='italic', family='sans-serif') 
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold', fontstyle='italic', family='sans-serif') 
plt.title('AUROC comparison across 20 models', fontsize=12, fontweight='bold', family='sans-serif') 
plt.legend(loc='lower right', fontsize='8', ncol=2)
plt.grid(True, linestyle=':', alpha=0.6)

# Lưu file kết quả dạng vector (SVG)
plt.tight_layout()
plt.savefig(output_path, format='svg', dpi=300)
print(f"✅ Đã lưu biểu đồ thành công tại: {output_path}")
plt.show()