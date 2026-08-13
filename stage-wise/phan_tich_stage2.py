import pandas as pd

df = pd.read_csv("your_3000_compounds_with_all_steps.csv")

total_initial = len(df)
actives_initial = df['label'].sum() # Tổng số chất Active thực tế trong 3025 mẫu

# Định nghĩa các bước đi qua tuần tự từng lớp phễu
steps = [
    ("1. Ban đầu (Initial)", pd.Series([True]*len(df))),
    ("2. Sau ML Consensus", df['pass_ml'] == 1),
    ("3. Sau ML + Toxicity", (df['pass_ml'] == 1) & (df['pass_toxicity'] == 1)),
    ("4. Sau ML + Tox + SA Score", (df['pass_ml'] == 1) & (df['pass_toxicity'] == 1) & (df['pass_sa'] == 1)),
    ("5. Sau Final Docking", (df['pass_ml'] == 1) & (df['pass_toxicity'] == 1) & (df['pass_sa'] == 1) & (df['pass_docking'] == 1))
]

results = []
for step_name, condition in steps:
    subset = df[condition]
    n_remaining = len(subset)
    n_actives = subset['label'].sum()
    n_inactives = n_remaining - n_actives
    
    # Tỷ lệ bảo toàn chất active so với ban đầu (%)
    active_recovery = (n_actives / actives_initial) * 100
    # Tỷ lệ làm giàu (Enrichment Factor - EF)
    baseline_ratio = actives_initial / total_initial
    current_ratio = n_actives / n_remaining if n_remaining > 0 else 0
    ef = current_ratio / baseline_ratio if baseline_ratio > 0 else 0
    
    results.append({
        "Workflow Stage": step_name,
        "Remaining Compounds": n_remaining,
        "Actives Retained": n_actives,
        "Active Recovery (%)": f"{active_recovery:.1f}%",
        "Enrichment Factor (EF)": f"{ef:.2f}x"
    })

df_report = pd.DataFrame(results)
print(df_report.to_string(index=False))