import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd
import seaborn as sns

# Đọc file dữ liệu
df = pd.read_csv("consensus_ad_combos.csv")

plt.figure(figsize=(8, 6))

# Vẽ scatter plot với ký hiệu "X"
scatter = sns.scatterplot(
    data=df,
    x="AUROC",
    y="AUPRC",
    hue="Model_Type",
    style="Model_Type",
    markers=["X"] * df["Model_Type"].nunique(),
    s=120,
    palette="Dark2",
    alpha=0.9,
)

# === THÊM CÁC ĐƯỜNG THRESHOLD ===
plt.axvline(x=0.8, color="#13578B", linestyle="--", linewidth=1.2)
plt.axhline(y=0.8, color="#942A7F", linestyle="-.", linewidth=1.2)

# === GHI CHÚ CHO "Consensus 6" ===
# Giả sử tên cột chứa tên mô hình/framework là cột đầu tiên (df.columns[0])
# Bạn hãy thay 'Framework' bằng tên cột chính xác trong file CSV của bạn nếu cần
target_col = df.columns[0] 

# Tìm dòng chứa từ khóa "Consensus" hoặc "6" (có thể đổi từ khóa theo đúng dữ liệu thực tế của bạn)
consensus_6_row = df[df[target_col].astype(str).str.contains("6", case=False, na=False)]

if not consensus_6_row.empty:
    for _, row in consensus_6_row.iterrows():
        x_val = row["AUROC"]
        y_val = row["AUPRC"]
        
        plt.annotate(
            "Consensus 6",
            (x_val, y_val),
            textcoords="offset points",
            xytext=(12, 12),  # Khoảng cách lệch của chữ so với điểm đánh dấu
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#171616",
            arrowprops=dict(
                arrowstyle="->",
                color="#161515",
                lw=1.5
            ),
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")]
        )
else:
    print("[CẢNH BÁO]: Không tìm thấy dòng chứa 'Consensus 6' hoặc số '6' trong file CSV để vẽ mũi tên.")

plt.xlabel("AUROC", fontweight="bold", fontstyle="italic", fontsize=12)
plt.ylabel("AUPRC", fontweight="bold", fontstyle="italic", fontsize=12)

plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("tradeoff_auroc.svg", format="svg", dpi=300)
plt.close()

print("Đã xuất file xxx.svg thành công!")