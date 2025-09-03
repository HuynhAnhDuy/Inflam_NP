import pandas as pd
import matplotlib.pyplot as plt

output_dir = "shap_scaffold_analysis_20250602_191119"

# Đọc dữ liệu
df = pd.read_csv(f'{output_dir}/feature_importance_mean.csv')
df_sorted = df.sort_values(by='Importance', ascending=False)
top_10 = df_sorted.head(20).copy()

# Thêm số thứ tự
top_10['Ranked_Feature'] = [f'({i+1}) {feat}' for i, feat in enumerate(top_10['Feature'])]

# Tạo biểu đồ
plt.figure(figsize=(5, 6))
bars = plt.barh(top_10['Ranked_Feature'], top_10['Importance'], color="#AC2126", edgecolor='black', alpha=0.8)

# Tính giới hạn trục X để chứa phần số
max_value = top_10['Importance'].max()
plt.xlim(0, max_value * 1.21)  # Thêm 15% bên phải để đủ chỗ cho số

# Thêm nhãn giá trị
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
             f'{bar.get_width():.4f}', va='center', ha='left', color='black', fontsize=11)

# Nhãn trục
plt.ylabel('MACSS keys fingerprints', fontsize=12, fontweight='bold', fontstyle='italic', family='sans-serif')
plt.xlabel('Mean absolute SHAP values', fontsize=12, fontweight='bold', fontstyle='italic', family='sans-serif')

# Đảo trục y và in đậm
plt.gca().invert_yaxis()
plt.gca().set_yticklabels(plt.gca().get_yticklabels())

# Xoay nhãn trục X
plt.xticks(rotation=45)

# === Vẽ đường viền bao ngoài toàn bộ plot (kể cả phần chữ) ===
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1)
    spine.set_edgecolor("black")

# Lưu biểu đồ
plt.tight_layout()
plt.savefig(f'{output_dir}/top_20_maccs.svg', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Completed Figure!")
