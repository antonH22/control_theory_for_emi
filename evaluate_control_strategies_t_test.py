import pandas as pd
from scipy.stats import ttest_1samp
import os

filename = "results_rnn_steps_mean.csv"
filepath1 = os.path.join("results_control_strategies", filename)
df_results = pd.read_csv(filepath1)

t_test_results = {"n_steps": [], "strategy": [], "t_statistic": [], "p_value": []}

n_steps = list(range(1, 13))

# Loop through each prediction length and strategy
for step in n_steps:
    for strategy in ["optimal control", "brute force", "max ac"]:
        column_name = f"{strategy} (n={step})"
        
        if column_name in df_results.columns:
            values = df_results[column_name] # List of mean wellbeing changes over all participants (per n_steps and strategy)
            t_stat, p_value = ttest_1samp(values, 0, alternative='less')  # One-tailed test (testing for improvement)
            t_test_results["n_steps"].append(step)
            t_test_results["strategy"].append(strategy)
            t_test_results["t_statistic"].append(t_stat)
            t_test_results["p_value"].append(p_value)

df_ttest = pd.DataFrame(t_test_results)

filepath2 = os.path.join("results_control_strategies", filename[:-4]+"_t_test.csv")
df_ttest.to_csv(filepath2, index=False)

print("T-test results saved successfully.")
print(df_ttest)
