import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, average_precision_score

# ====== Cấu hình đường dẫn các thư mục mô hình ======
base_dir = r"D:\Andy\Inflam_NP\Predictive_models\Prob_InFlam_full"

model_dirs = {
    'RF': os.path.join(base_dir, "Prob_RF"),
    'XGB': os.path.join(base_dir, "Prob_XGB"),    
    'LGBM': os.path.join(base_dir, "Prob_LGBM"),  
    'BiLSTM': os.path.join(base_dir, "Prob_BiLSTM") 
}

features_to_include = ['ecfp', 'maccs', 'rdkit', 'estate', 'phychem']
output_path = "AUPRC_plot.svg"  # Đổi tên file đầu ra thành AUPRC
colormap_name = 'tab20'  # Dùng bảng màu 20 màu cho đủ các mô hình x features
# ====================================================

prc_data = []
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
            
            # Tính Precision, Recall và diện tích AUPRC
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            # Lưu ý: Trong PRC, trục hoành là Recall, trục tung là Precision. Khi tính auc(x, y), x là recall, y là precision.
            auprc = auc(recall, precision)
            
            label_name = f"{model_name} + {feat.upper()}"
            prc_data.append((label_name, recall, precision, auprc))

# Sắp xếp theo AUPRC giảm dần để thứ tự hiển thị trong Legend trực quan hơn
prc_data_sorted = sorted(prc_data, key=lambda x: x[3], reverse=True)

# Lấy colormap
cmap = plt.get_cmap(colormap_name)
colors = [cmap(i % cmap.N) for i in range(len(prc_data_sorted))]

# Vẽ từng đường Precision-Recall
for idx, (model_label, recall, precision, auprc) in enumerate(prc_data_sorted):
    plt.plot(recall, precision,
             label=f'{model_label} (AUPRC={auprc:.3f})',
             color=colors[idx],
             linewidth=1.5,
             linestyle='-')

# Đối với biểu đồ PRC, đường tham chiếu ngẫu nhiên (Baseline) thường là tỷ lệ mẫu dương trên tổng số (hoặc đường ngang tương ứng với tỷ lệ positive class nếu cần, tuy nhiên thường có thể bỏ qua hoặc vẽ đường ngang tỉ lệ positive. Ở đây dùng đường ngang mức random tương ứng với tỷ lệ baseline nếu muốn, hoặc tạm bỏ đường chéo ROC). 
# Tạm thời vẽ đường ngang mức positive rate tùy thuộc vào dữ liệu, hoặc vẽ đường nét đứt ngang ở mức 0.5 nếu bài toán cân bằng. Ở đây ta có thể ẩn hoặc để trống đường random cũ.

# Tùy chỉnh biểu đồ
plt.xlabel('Recall', fontsize=12, fontweight='bold', fontstyle='italic', family='sans-serif') 
plt.ylabel('Precision', fontsize=12, fontweight='bold', fontstyle='italic', family='sans-serif') 
plt.title('AUPRC comparison across 20 models', fontsize=12, fontweight='bold', family='sans-serif') 
plt.legend(loc='lower left', fontsize='8', ncol=2)
plt.grid(True, linestyle=':', alpha=0.6)
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])

# Lưu file kết quả dạng vector (SVG)
plt.tight_layout()
plt.savefig(output_path, format='svg', dpi=300)
print(f"✅ Đã lưu biểu đồ AUPRC thành công tại: {output_path}")
plt.show()