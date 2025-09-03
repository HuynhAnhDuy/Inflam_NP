import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Load your data
df = pd.read_csv('irritation_att_new_all_phychem_boxplot.csv')

# List of dependent variables
dependent_vars = ['MolWt', 'TPSA', 'HeavyAtomCount']

# Specify the independent variable
independent_var = 'Group'

# Initialize lists to store results
results = {}
anova_results = []
tukey_results = []

# Perform ANOVA and Tukey's HSD for each dependent variable
for dep_var in dependent_vars:
    print(f"Performing ANOVA for {dep_var}...")
    
    # Fit the model
    model = ols(f'{dep_var} ~ C({independent_var})', data=df).fit()
    
    # Perform the ANOVA
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_results.append(anova_table)
    
    # Check if ANOVA is significant
    if anova_table['PR(>F)'][0] < 0.05:
        tukey = pairwise_tukeyhsd(endog=df[dep_var], groups=df[independent_var], alpha=0.05)
        tukey_summary = tukey.summary().as_csv()
        tukey_results.append((dep_var, tukey_summary))
    else:
        tukey_results.append((dep_var, 'No significant differences found.'))

    print("\n")

# Save ANOVA results to CSV
anova_results_df = pd.concat(anova_results, keys=dependent_vars, names=['Dependent Variable'])
anova_results_df.to_csv('irritation_phychem_anova_results.csv')

# Save Tukey's HSD results to CSV
with open('irritation_phychem_tukey_results.csv', 'w') as f:
    for dep_var, result in tukey_results:
        f.write(f"\n\n{dep_var}\n")
        f.write(result if isinstance(result, str) else result)
