import pandas as pd
from scipy import stats

# Load your data
df = pd.read_csv('sensitization.csv')

# List of dependent variables
dependent_vars = ['ACC_test', 'MCC_test','roc_auc_test', 'BACC_test','Precision_test','Sens_recall_test','Spec_test']

# Specify the independent variable
independent_var = 'model'

# Initialize a dictionary to store Levene's Test results
levene_results = {}

# Perform Levene's Test for each dependent variable
for var in dependent_vars:
    print(f"Performing Levene's Test for {var}...")
    
    # Group by the independent variable and collect data for Levene's Test
    grouped_data = [group[var].values for name, group in df.groupby(independent_var)]
    
    # Perform Levene's Test
    try:
        stat, p_value = stats.levene(*grouped_data)
        levene_results[var] = {'Levene Statistic': stat, 'p-value': p_value}
        print(f"Levene's Test for {var}: Statistic={stat}, p-value={p_value}")
    except ValueError as e:
        print(f"Levene's Test failed for {var}: {e}")
        levene_results[var] = {'Levene Statistic': None, 'p-value': None}

# Save Levene's Test results to CSV
levene_results_df = pd.DataFrame(levene_results).T.reset_index()
levene_results_df.rename(columns={'index': 'Dependent Variable'}, inplace=True)
levene_results_df.to_csv('sensitization_levene_test_results.csv', index=False)
