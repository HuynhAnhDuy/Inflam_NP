import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("consensus_ad_combos.csv")

plt.figure(figsize=(8, 5))

# Đổi trục: x là AUPRC, y là AUROC
scatter = sns.scatterplot(
    data=df,
    x="AUROC",
    y="AUPRC",
    hue="Model_Type",
    s=100,
    palette="Dark2",
    alpha=0.9,
)

# === THÊM CÁC ĐƯỜNG THRESHOLD (NGƯỠNG MÀU ĐỎ CHO AUPRC = 0.7 VÀ AUROC = 0.7) ===
plt.axvline(
    x=0.8,
    color="#13578B",
    linestyle="--",
    linewidth=1.2,
    label="AUROC threshold = 0.8",
)
plt.axhline(
    y=0.8,
    color="#942A7F",
    linestyle="-.",
    linewidth=1.2,
    label="AUPRC threshold = 0.8",
)

plt.xlabel(
    "AUROC",
    fontweight="bold",
    fontstyle="italic",
    fontsize=12,
)
plt.ylabel(
    "AUPRC",
    fontweight="bold",
    fontstyle="italic",
    fontsize=12,
)

plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("tradeoff_auroc.svg", format="svg", dpi=300)