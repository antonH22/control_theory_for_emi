import pandas as pd
from scipy.stats import ttest_1samp
import os
import numpy as np

### Statistical analysis of control strategies (A.1, Table 5, 6)

filename = "results_control_strategies_offline_bias_corrected.csv"
filepath1 = os.path.join("results_replicated", filename)
df_results = pd.read_csv(filepath1)

t_test_results = {"n_steps": [], "strategy": [], "t_statistic": [], "p_value": [], "cohens_d": []}
t_test_results_acc = {"n_steps": [], "strategy": [], "t_statistic": [], "p_value": []}

n_steps = list(range(1, 13))
strategies = ["optimal control", "brute force", "max ac", "no emi"]

def cohens_d_one_sample(data):
    mean_diff = np.nanmean(data)
    std_dev = np.nanstd(data, ddof=1)
    return mean_diff / std_dev

# Loop through each prediction length and strategy
for strategy in strategies:
    for step in n_steps:
        column_name = f"{strategy} (n={step})"
        values = df_results[column_name].dropna() # List of mean wellbeing changes over all participants (per n_steps and strategy)
        t_stat, p_value = ttest_1samp(values, 0, alternative='greater')  # One-tailed test (testing for improvement)
        d_value = cohens_d_one_sample(values)  # Compute Cohen's d
        t_test_results["n_steps"].append(step)
        t_test_results["strategy"].append(strategy)
        t_test_results["t_statistic"].append(t_stat)
        t_test_results["p_value"].append(p_value)
        t_test_results["cohens_d"].append(d_value)

df_ttest = pd.DataFrame(t_test_results)
filepath2 = os.path.join("results_control_strategies", filename[:-4]+"_t_test.csv")
df_ttest.to_csv(filepath2, index=False)

print("T-test results saved successfully.")
print(df_ttest)
