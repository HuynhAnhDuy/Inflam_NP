import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd
import seaborn as sns

# Đọc file dữ liệu
df = pd.read_csv("consensus_ad_combos.csv")

plt.figure(figsize=(6, 4))

# Vẽ scatter plot với ký hiệu "X"
scatter = sns.scatterplot(
    data=df,
    x="MCC",
    y="Brier_Score",
    hue="Model_Type",
    style="Model_Type",
    markers=["X"] * df["Model_Type"].nunique(),
    s=120,
    palette="Dark2",
    alpha=0.9,
)

# === THÊM CÁC ĐƯỜNG THRESHOLD ===
plt.axvline(x=0.5, color="#13578B", linestyle="--", linewidth=1.2)
plt.axhline(y=0.25, color="#942A7F", linestyle="-.", linewidth=1.2)

plt.xlabel("MCC", fontweight="bold", fontstyle="italic", fontsize=12)
plt.ylabel("Brier score", fontweight="bold", fontstyle="italic", fontsize=12)

plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("tradeoff_mcc.svg", format="svg", dpi=300)
plt.close()

print("Đã xuất file xxx.svg thành công!")