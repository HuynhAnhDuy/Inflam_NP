import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Bước 1: Load data
csv_path = 'InFlam_full_all_metrics.csv'  # Đường dẫn file của bạn
df = pd.read_csv(csv_path)

# Đặt cột 'Model' làm index nếu có
if 'Model' in df.columns:
    df.set_index('Model', inplace=True)

# Chỉ giữ 4 cột mong muốn
metrics_to_plot = ["MCC", "Accuracy", "Sensitivity", "Specificity"]
df_subset = df[metrics_to_plot].copy()

# Bước 2: Xử lý chuỗi "mean ± std" để lấy giá trị số (mean) cho tất cả các cột
for col in df_subset.columns:
    if df_subset[col].dtype == object:
        df_subset[col] = df_subset[col].astype(str).str.split('±').str[0].str.strip().astype(float)

# Bước 3: Sắp xếp các hàng theo giá trị của cột "MCC" giảm dần
if 'MCC' in df_subset.columns:
    df_subset = df_subset.sort_values(by='MCC', ascending=False)

# Bước 4: Thiết lập figure
plt.figure(figsize=(9, len(df_subset) * 0.4))  # Tự động điều chỉnh chiều cao theo số lượng mô hình

# Bước 5: Vẽ heatmap
ax = sns.heatmap(df_subset,
                 cmap='RdYlBu',        # Bảng màu: Đỏ/Vàng → Xanh
                 annot=True,          # Hiển thị giá trị lên ô
                 fmt=".3f",           # Định dạng 3 chữ số thập phân
                 linewidths=0.5,
                 linecolor="black",
                 cbar_kws={'label': 'Metric values'})

# Bước 6: Định dạng màu sắc và kích thước text tự động theo giá trị
for text in ax.texts:
    val = float(text.get_text())
    text.set_color('white' if (val < 0.3 or val > 0.85) else 'black')  
    text.set_fontsize(10)

# Bước 7: Nhãn trục và định dạng
plt.yticks(rotation=0, fontsize=11, family='sans-serif')
plt.xticks(rotation=45, ha='right', fontsize=11, family='sans-serif')
plt.xlabel('') 
plt.ylabel('') 

# Bước 9: Lưu và hiển thị
plt.tight_layout()
plt.savefig('heatmap_sorted_MCC.svg', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Đã xử lý dữ liệu, sắp xếp theo MCC và lưu biểu đồ heatmap thành công!")