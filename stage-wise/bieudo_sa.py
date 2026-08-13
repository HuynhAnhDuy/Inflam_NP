import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the analysis report directly
df = pd.read_csv("screening_sa_analysis.csv")

# 2. Calculate totals and percentages
df["Kept_Total"] = df["Kept_Positive"] + df["Kept_Negative"]
initial_total = df["Kept_Total"].iloc[-1] + df["Dropped_Positive"].iloc[-1] + df["Dropped_Negative"].iloc[-1]

# Tỷ lệ mẫu dương so với tổng ban đầu (%)
df["Kept_Positive_Ratio"] = (df["Kept_Positive"] / initial_total) * 100

# Tỷ lệ tổng số mẫu được giữ lại so với tổng ban đầu (%)
df["Total_Kept_Ratio"] = (df["Kept_Total"] / initial_total) * 100

# 3. Set up the plotting environment (2 subplots vertically)
sns.set_theme(style="whitegrid", context="talk")
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 13), sharex=True, gridspec_kw={'height_ratios': [3, 1.2]})

# --- Subplot 1: Stacked Bar Chart (Kept vs Dropped) ---
ax1.bar(df["SA_Threshold"], df["Kept_Positive"], label="Kept positive", color="#4C72B0")
ax1.bar(df["SA_Threshold"], df["Kept_Negative"], bottom=df["Kept_Positive"], label="Kept negative", color="#55A868")
ax1.bar(df["SA_Threshold"], df["Dropped_Positive"], bottom=df["Kept_Positive"] + df["Kept_Negative"], label="Dropped positive", color="#C44E52", alpha=0.7)
ax1.bar(df["SA_Threshold"], df["Dropped_Negative"], bottom=df["Kept_Positive"] + df["Kept_Negative"] + df["Dropped_Positive"], label="Dropped negative", color="#8172B3", alpha=0.7)

ax1.set_ylabel("Number of samples", fontsize=14, fontweight="bold", fontstyle="italic")
ax1.legend(loc="upper left", bbox_to_anchor=(1, 1))

# --- Subplot 2: Line Chart for Ratios with Dual Annotations ---
ax2.plot(df["SA_Threshold"], df["Kept_Positive_Ratio"], marker='o', linestyle='-', linewidth=2.5, color="#D90258", label="Kept positive ratio (vs initial) (%)")
ax2.plot(df["SA_Threshold"], df["Total_Kept_Ratio"], marker='s', linestyle='--', linewidth=2.5, color="#1F11E2", label="Total kept ratio (vs initial) (%)")

# --- ĐIỀU CHỈNH VỊ TRÍ NHÃN (ANNOTATIONS) ---

# 1. Kept Positive Ratio: Chuyển xuống DƯỚI điểm marker (xytext=(0, -18))
for i, txt in enumerate(df["Kept_Positive_Ratio"]):
    ax2.annotate(f"{txt:.1f}%", 
                 (df["SA_Threshold"][i], df["Kept_Positive_Ratio"][i]),
                 textcoords="offset points", 
                 xytext=(0, -18), 
                 ha='left', 
                 fontsize=10, 
                 fontweight='bold',
                 color="#101010")

# 2. Total Kept Ratio: Chuyển lên TRÊN điểm marker (xytext=(0, 12))
for i, txt in enumerate(df["Total_Kept_Ratio"]):
    ax2.annotate(f"{txt:.1f}%", 
                 (df["SA_Threshold"][i], df["Total_Kept_Ratio"][i]),
                 textcoords="offset points", 
                 xytext=(0, 12), 
                 ha='right', 
                 fontsize=10, 
                 fontweight='bold',
                 color="#0C0C0D")

ax2.set_xlabel("SA score threshold", fontsize=14, fontweight="bold", fontstyle="italic")
ax2.set_ylabel("Percentage (%)", fontsize=14, fontweight="bold", fontstyle="italic")
ax2.set_xticks(range(1, 11))
ax2.set_ylim(-5, 115) # Mở rộng trục y để chứa nhãn phía dưới rõ hơn
ax2.legend(loc="upper left", bbox_to_anchor=(1, 1))

plt.tight_layout()

# 4. Save as SVG format
output_filename = "stacked_bar_with_dual_ratios.svg"
plt.savefig(output_filename, format="svg")
print(f"Combined chart saved successfully as {output_filename}.")