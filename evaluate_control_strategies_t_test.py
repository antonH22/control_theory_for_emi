import pandas as pd
from scipy.stats import ttest_1samp
import os

filename = "results_rnn_online_step.csv"
filepath1 = os.path.join("results_control_strategies", filename)
df_results = pd.read_csv(filepath1)

t_test_results = {"n_steps": [], "strategy": [], "t_statistic": [], "p_value": []}
t_test_results_acc = {"n_steps": [], "strategy": [], "t_statistic": [], "p_value": []}

n_steps = list(range(1, 13))
strategies = ["optimal control", "brute force", "max ac", "no emi"]

# Loop through each prediction length and strategy
for strategy in strategies:
    for step in n_steps:
        column_name = f"{strategy} (n={step})"
        values = df_results[column_name].dropna() # List of mean wellbeing changes over all participants (per n_steps and strategy)
        t_stat, p_value = ttest_1samp(values, 0, alternative='less')  # One-tailed test (testing for improvement)
        t_test_results["n_steps"].append(step)
        t_test_results["strategy"].append(strategy)
        t_test_results["t_statistic"].append(t_stat)
        t_test_results["p_value"].append(p_value)

# Accumulate the values over steps for each strategy and then perform the t-test on the accumulated data 
for strategy in strategies:
    accumulated_values = []
    for step in n_steps:
        column_name = f"{strategy} (n={step})" 
        accumulated_values.extend(df_results[column_name].dropna())
        t_stat_acc, p_value_acc = ttest_1samp(accumulated_values, 0, alternative='less')  # One-tailed test (testing for improvement)
        t_test_results_acc["n_steps"].append(step)
        t_test_results_acc["strategy"].append(strategy)
        t_test_results_acc["t_statistic"].append(t_stat_acc)
        t_test_results_acc["p_value"].append(p_value_acc)

df_ttest = pd.DataFrame(t_test_results)
filepath2 = os.path.join("results_control_strategies", filename[:-4]+"_t_test.csv")
df_ttest.to_csv(filepath2, index=False)

print("T-test results saved successfully.")
print(df_ttest)
