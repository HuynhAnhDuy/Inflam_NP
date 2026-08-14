import pandas as pd

# 1. Đọc file dữ liệu
df = pd.read_csv("full_processed_data_SA2_affinity.csv")

total_initial = len(df)
actives_initial = int(df['True label'].sum())

# ==========================================================
# PHẦN 1: TÍNH ĐÓNG GÓP ĐỘC LẬP (Independent Contribution)
# ==========================================================
ind_ml_removed = (df['pass_ml'] != 'yes').sum()
ind_sa2_removed = (df['pass_sa_2'] != 'yes').sum()
ind_tox_removed = (df['pass_toxicity'] != 'yes').sum()
ind_dock_removed = (df['affinity (kcal/mol)'] >= -7.0).sum()

df_independent = pd.DataFrame([
    {
        "Filter component": "ML consensus",
        "Total removed": ind_ml_removed,
        "Independent rejection rate (%)": f"{(ind_ml_removed / total_initial) * 100:.1f}%"
    },
    {
        "Filter component": "SA threshold of 2",
        "Total removed": ind_sa2_removed,
        "Independent rejection rate (%)": f"{(ind_sa2_removed / total_initial) * 100:.1f}%"
    },
    {
        "Filter component": "Toxicity filter",
        "Total removed": ind_tox_removed,
        "Independent rejection rate (%)": f"{(ind_tox_removed / total_initial) * 100:.1f}%"
    },
    {
        "Filter component": "Final docking (affinity < -7.0)",
        "Total removed": ind_dock_removed,
        "Independent rejection rate (%)": f"{(ind_dock_removed / total_initial) * 100:.1f}%"
    }
])

print("--- ĐÓNG GÓP ĐỘC LẬP (Independent Contribution) ---")
print(df_independent.to_string(index=False))
print("\n" + "="*70 + "\n")

# Lưu kết quả độc lập ra CSV
df_independent.to_csv("independent_contribution_report.csv", index=False)


# ==========================================================
# PHẦN 2: TÍNH ĐÓNG GÓP TUẦN TỰ (Sequential Contribution)
# ==========================================================

# --- HƯỚNG 1: ML -> SA2 -> Toxicity -> Docking ---
steps_ml_first = [
    ("1. Initial", pd.Series([True]*len(df))),
    ("2. After ML consensus", df['pass_ml'] == 'yes'),
    ("3. After ML + SA2", (df['pass_ml'] == 'yes') & (df['pass_sa_2'] == 'yes')),
    ("4. After ML + SA2 + toxicity", (df['pass_ml'] == 'yes') & (df['pass_sa_2'] == 'yes') & (df['pass_toxicity'] == 'yes')),
    ("5. After final docking", (df['pass_ml'] == 'yes') & (df['pass_sa_2'] == 'yes') & (df['pass_toxicity'] == 'yes') & (df['affinity (kcal/mol)'] < -7.0))
]

results_ml_first = []
for step_name, condition in steps_ml_first:
    subset = df[condition]
    n_remaining = len(subset)
    n_actives = int(subset['True label'].sum())
    active_retention = (n_actives / actives_initial) * 100 if actives_initial > 0 else 0
    n_inactives = n_remaining - n_actives
    active_ratio = (n_actives / n_remaining * 100) if n_remaining > 0 else 0
    
    results_ml_first.append({
        "Workflow stage": step_name,
        "Remaining compounds": n_remaining,
        "Actives retained": n_actives,
        "Active retention rate (%)": f"{active_retention:.1f}%",
        "Inactives left": n_inactives,
        "Active ratio (%)": f"{active_ratio:.1f}%"
    })

df_seq_ml = pd.DataFrame(results_ml_first)
print("--- HƯỚNG 1: SEQUENTIAL REPORT (ML First) ---")
print(df_seq_ml.to_string(index=False))
print("\n" + "="*70 + "\n")
df_seq_ml.to_csv("sequential_report_ML_first.csv", index=False)


# --- HƯỚNG 2: SA2 -> Toxicity -> ML -> Docking ---
steps_sa2_first = [
    ("1. Initial", pd.Series([True]*len(df))),
    ("2. After SA2", df['pass_sa_2'] == 'yes'),
    ("3. After SA2 + toxicity", (df['pass_sa_2'] == 'yes') & (df['pass_toxicity'] == 'yes')),
    ("4. After SA2 + toxicity + ML", (df['pass_sa_2'] == 'yes') & (df['pass_toxicity'] == 'yes') & (df['pass_ml'] == 'yes')),
    ("5. After final docking", (df['pass_sa_2'] == 'yes') & (df['pass_toxicity'] == 'yes') & (df['pass_ml'] == 'yes') & (df['affinity (kcal/mol)'] < -7.0))
]

results_sa2_first = []
for step_name, condition in steps_sa2_first:
    subset = df[condition]
    n_remaining = len(subset)
    n_actives = int(subset['True label'].sum())
    active_retention = (n_actives / actives_initial) * 100 if actives_initial > 0 else 0
    n_inactives = n_remaining - n_actives
    active_ratio = (n_actives / n_remaining * 100) if n_remaining > 0 else 0
    
    results_sa2_first.append({
        "Workflow stage": step_name,
        "Remaining compounds": n_remaining,
        "Actives retained": n_actives,
        "Active retention rate (%)": f"{active_retention:.1f}%",
        "Inactives left": n_inactives,
        "Active ratio (%)": f"{active_ratio:.1f}%"
    })

df_seq_sa2 = pd.DataFrame(results_sa2_first)
print("--- HƯỚNG 2: SEQUENTIAL REPORT (SA2 First) ---")
print(df_seq_sa2.to_string(index=False))
df_seq_sa2.to_csv("sequential_report_SA2_first.csv", index=False)