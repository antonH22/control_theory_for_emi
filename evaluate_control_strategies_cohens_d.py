import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
import os
from ctrl import utils

def cohens_d_one_sample(data):
    mean_diff = np.nanmean(data)
    std_dev = np.nanstd(data, ddof=1)
    return mean_diff / std_dev

filename = "results_rnn_online_step.csv"
filepath1 = os.path.join("results_control_strategies", filename)
df_results = pd.read_csv(filepath1)

cohens_d_results = {"n_steps": [], "strategy": [], "cohens_d": []}
cohens_d_results_acc = {"n_steps": [], "strategy": [], "cohens_d": []}

n_steps = list(range(1, 13))
strategies = ["optimal control", "brute force", "max ac", "no emi"]
# Loop through each prediction length and strategy

for strategy in strategies:
    for step in n_steps:
        column_name = f"{strategy} (n={step})"
        values = df_results[column_name] # List of mean wellbeing changes over all participants (per n_steps and strategy)
        d_value = cohens_d_one_sample(values)  # Compute Cohen's d
        cohens_d_results["n_steps"].append(step)
        cohens_d_results["strategy"].append(strategy)
        cohens_d_results["cohens_d"].append(d_value)

# Accumulate the values over steps for each strategy and then compute the cohens_d on the accumulated data 
for strategy in strategies:
    accumulated_values = []
    for step in n_steps:
        column_name = f"{strategy} (n={step})" 
        accumulated_values.extend(df_results[column_name].dropna())
        d_value_acc = cohens_d_one_sample(accumulated_values)  # Compute Cohen's d
        cohens_d_results_acc["n_steps"].append(step)
        cohens_d_results_acc["strategy"].append(strategy)
        cohens_d_results_acc["cohens_d"].append(d_value_acc)

# Convert results to DataFrame
df_cohens_d = pd.DataFrame(cohens_d_results)

filepath2 = os.path.join("results_control_strategies", filename[:-4]+"_cohens_d.csv")
df_cohens_d.to_csv(filepath2, index=False)

print("Cohens_d results saved successfully.")
print(df_cohens_d)