import pandas as pd
from scipy import stats
import numpy as np

# === cấu hình ===
INPUT_CSV  = "InFlam_full_all_metrics_raw.csv"   # đổi nếu cần
OUTPUT_CSV = "InFlam_full_all_metrics_raw_shapiro.csv"
ALPHA = 0.05

# Load data
df = pd.read_csv(INPUT_CSV)

# Các biến cần kiểm định
dependent_vars = ['Accuracy', 'MCC', 'Sensitivity', 'Specificity', 'AUROC']

rows = []
for var in dependent_vars:
    print(f"Performing Shapiro-Wilk Test for {var}...")

    # Ép kiểu numeric và loại NaN
    s = pd.to_numeric(df[var], errors='coerce').dropna()
    n = int(s.shape[0])

    stat = np.nan
    p_value = np.nan
    assessment = ""

    if n < 3:
        assessment = "Insufficient data (n<3)"
    elif s.nunique() == 1:
        assessment = "Constant values (cannot test)"
    else:
        try:
            stat, p_value = stats.shapiro(s)
            assessment = "Normal" if p_value >= ALPHA else "Non-normal"
        except Exception as e:
            assessment = f"Error: {type(e).__name__}"

    print(f"Shapiro-Wilk Test for {var}: Statistic={stat}, p-value={p_value} -> {assessment}")

    rows.append({
        "Dependent Variable": var,
        "N": n,
        "Alpha": ALPHA,
        "Shapiro-Wilk Statistic": stat,
        "p-value": p_value,
        "Assessment": assessment
    })

# Lưu kết quả
shapiro_results_df = pd.DataFrame(rows)
# (tuỳ chọn) làm đẹp số
shapiro_results_df["Shapiro-Wilk Statistic"] = shapiro_results_df["Shapiro-Wilk Statistic"].round(6)
shapiro_results_df["p-value"] = shapiro_results_df["p-value"].round(10)

shapiro_results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved Shapiro results with Assessment → {OUTPUT_CSV}")
