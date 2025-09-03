import pandas as pd
from scipy import stats
import scikit_posthocs as sp

# Load your data
df = pd.read_csv('InFlam_full_all_metrics_raw.csv')

# List of dependent variables
dependent_vars = ['Accuracy', 'MCC', 'Sensitivity','Specificity',"AUROC"]

# Specify the independent variable
independent_var = 'Model'

# Initialize dictionaries to store Kruskal-Wallis and Dunn's Test results
kruskal_results = {}
dunn_results = {}

# Perform Kruskal-Wallis H-Test and Dunn's Test for each dependent variable
for var in dependent_vars:
    print(f"Performing Kruskal-Wallis H-Test for {var}...")
    
    # Group by the independent variable and collect data for Kruskal-Wallis Test
    grouped_data = [group[var].values for name, group in df.groupby(independent_var)]
    
    # Perform Kruskal-Wallis H-Test
    stat, p_value = stats.kruskal(*grouped_data)
    
    # Determine significance
    significance = 'Significant differnence' if p_value < 0.05 else 'Not Significant'
    
    # Store Kruskal-Wallis results
    kruskal_results[var] = {
        'Kruskal-Wallis H Statistic': stat,
        'p-value': p_value,
        'Significance': significance
    }
    
    print(f"Kruskal-Wallis H-Test for {var}: Statistic={stat:.4f}, p-value={p_value:.4f}, {significance}")
    
    # Perform Dunn's Test if Kruskal-Wallis test is significant
    if p_value < 0.05:
        print(f"Performing Dunn's Test for {var}...")
        
        # Combine data into a DataFrame for Dunn's test
        dunn_df = df[[independent_var, var]].copy()
        
        # Perform Dunn's Test
        dunn_result = sp.posthoc_dunn(dunn_df, val_col=var, group_col=independent_var)
        
        # Store Dunn's Test results
        dunn_results[var] = dunn_result
        print(f"Dunn's Test for {var}:\n{dunn_result}")

# Save Kruskal-Wallis H-Test results to CSV
kruskal_results_df = pd.DataFrame(kruskal_results).T.reset_index()
kruskal_results_df.rename(columns={'index': 'Dependent Variable'}, inplace=True)
kruskal_results_df.to_csv('InFlam_full_kruskal_wallis.csv', index=False)

# Save Dunn's Test results to CSV (only if there are results)
if dunn_results:
    for var, result in dunn_results.items():
        result.to_csv(f'InFlam_full_dunn_{var}.csv', index=True)
