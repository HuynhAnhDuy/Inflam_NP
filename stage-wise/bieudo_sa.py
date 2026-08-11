import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the analysis report
try:
    df = pd.read_csv("screening_sa_analysis.csv")
except FileNotFoundError:
    # Fallback sample data if file is not found
    data = {
        "SA_Threshold": range(1, 11),
        "Kept_Positive": [10, 30, 80, 150, 300, 450, 600, 700, 800, 900],
        "Kept_Negative": [20, 50, 120, 250, 500, 800, 1100, 1300, 1500, 1600],
        "Dropped_Positive": [890, 870, 820, 750, 600, 450, 300, 200, 100, 0],
        "Dropped_Negative": [1580, 1550, 1480, 1350, 1100, 800, 500, 300, 100, 0]
    }
    df = pd.DataFrame(data)

# 2. Calculate totals and percentages
df["Kept_Total"] = df["Kept_Positive"] + df["Kept_Negative"]
initial_total = df["Kept_Total"].iloc[-1] + df["Dropped_Positive"].iloc[-1] + df["Dropped_Negative"].iloc[-1]

# Tỷ lệ mẫu dương trên tổng số giữ lại (%)
df["Kept_Positive_Ratio"] = (df["Kept_Positive"] / df["Kept_Total"].replace(0, 1)) * 100

# Tỷ lệ tổng số mẫu được giữ lại so với tổng ban đầu (%)
df["Total_Kept_Ratio"] = (df["Kept_Total"] / initial_total) * 100

# 3. Set up the plotting environment (2 subplots vertically)
sns.set_theme(style="whitegrid", context="talk")
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 13), sharex=True, gridspec_kw={'height_ratios': [3, 1.2]})

# --- Subplot 1: Stacked Bar Chart (Kept vs Dropped) ---
ax1.bar(df["SA_Threshold"], df["Kept_Positive"], label="Kept Positive", color="#4C72B0")
ax1.bar(df["SA_Threshold"], df["Kept_Negative"], bottom=df["Kept_Positive"], label="Kept Negative", color="#55A868")
ax1.bar(df["SA_Threshold"], df["Dropped_Positive"], bottom=df["Kept_Positive"] + df["Kept_Negative"], label="Dropped Positive", color="#C44E52", alpha=0.6)
ax1.bar(df["SA_Threshold"], df["Dropped_Negative"], bottom=df["Kept_Positive"] + df["Kept_Negative"] + df["Dropped_Positive"], label="Dropped Negative", color="#8172B3", alpha=0.6)

ax1.set_title("Dataset Distribution and Ratios by SA Threshold", fontsize=18, fontweight='bold')
ax1.set_ylabel("Number of Samples", fontsize=14)
ax1.legend(loc="upper left", bbox_to_anchor=(1, 1))

# --- Subplot 2: Line Chart for Ratios with Dual Annotations ---
ax2.plot(df["SA_Threshold"], df["Kept_Positive_Ratio"], marker='o', linestyle='-', linewidth=2.5, color="#D95F02", label="Kept Positive Ratio (%)")
ax2.plot(df["SA_Threshold"], df["Total_Kept_Ratio"], marker='s', linestyle='--', linewidth=2.5, color="#7570B3", label="Total Kept Ratio (%)")

# Annotate Kept Positive Ratio values (Placed above the marker)
for i, txt in enumerate(df["Kept_Positive_Ratio"]):
    ax2.annotate(f"{txt:.1f}%", 
                 (df["SA_Threshold"][i], df["Kept_Positive_Ratio"][i]),
                 textcoords="offset points", 
                 xytext=(0, 10), 
                 ha='center', 
                 fontsize=9, 
                 fontweight='bold',
                 color="#D95F02")

# Annotate Total Kept Ratio values (Placed below the marker to avoid overlapping)
for i, txt in enumerate(df["Total_Kept_Ratio"]):
    ax2.annotate(f"{txt:.1f}%", 
                 (df["SA_Threshold"][i], df["Total_Kept_Ratio"][i]),
                 textcoords="offset points", 
                 xytext=(0, -15), 
                 ha='center', 
                 fontsize=9, 
                 fontweight='bold',
                 color="#7570B3")

ax2.set_xlabel("SA Score Threshold", fontsize=14)
ax2.set_ylabel("Percentage (%)", fontsize=14)
ax2.set_xticks(range(1, 11))
ax2.set_ylim(-5, 115) # Mở rộng trục y xuống một chút để hiển thị nhãn phía dưới rõ hơn
ax2.axhline(50, color='gray', linestyle=':', alpha=0.7, label='50% Reference')
ax2.legend(loc="upper left", bbox_to_anchor=(1, 1))

plt.tight_layout()

# 4. Save as SVG format
output_filename = "stacked_bar_with_dual_ratios.svg"
plt.savefig(output_filename, format="svg")
print(f"Combined chart saved successfully as {output_filename}.")