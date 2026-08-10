import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("consensus_ad_combos_evaluation_summary.csv")

plt.figure(figsize=(9, 6))

# Vẽ scatter plot (không hiển thị N_Samples)
scatter = sns.scatterplot(
    data=df,
    x="MCC",
    y="Brier_Score",
    hue="Model_Type",
    s=100,
    palette="viridis",
    alpha=0.85,
)

# === TỰ ĐỘNG TÌM VÀ GHI CHÚ TOP 2 MÔ HÌNH TỐT NHẤT ===
df["Score_Rank"] = df["MCC"] - df["Brier_Score"]
top2_models = df.sort_values(by="Score_Rank", ascending=False).head(2)

# Sửa lỗi thừa dấu ngoặc ở enumerate
for i, (index, row) in enumerate(top2_models.iterrows()):
  x_val = row["MCC"]
  y_val = row["Brier_Score"]

  # Chỉ lấy tên Framework, không ghi chú single/combo
  label_text = f"{row['Framework']}"

  # Luôn đặt chú thích nằm bên phải của point
  plt.annotate(
      label_text,
      xy=(x_val, y_val),
      xytext=(10, 0),  # Dịch sang phải 10 points, giữ nguyên chiều cao
      textcoords="offset points",
      fontsize=9,
      fontweight="bold",
      color="crimson",
      ha="left",  # Căn trái để chữ chạy sang phải từ điểm mốc
      va="center",  # Căn giữa theo chiều dọc với điểm
      path_effects=[
          pe.withStroke(linewidth=2.5, foreground="white")
      ],  # Viền trắng giúp dễ đọc
  )

plt.title(
    "Trade-off analysis across 64 configurations",
    fontsize=12,
    fontweight="bold",
)

plt.xlabel(
    "MCC (Higher is better)",
    fontweight="bold",
    fontstyle="italic",
    fontsize=12,
)
plt.ylabel(
    "Brier score (Lower is better)",
    fontweight="bold",
    fontstyle="italic",
    fontsize=12,
)

plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("models_tradeoff_comparison.svg", format="svg", dpi=300)
plt.show()