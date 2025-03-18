import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define file paths
filepath1 = os.path.join("results_control_strategies", "opt_ctrl_lds.csv")
filepath2 = os.path.join("results_control_strategies", "brute_force_lds.csv")
filepath3 = os.path.join("results_control_strategies", "max_ac_lds.csv")

# Read CSVs into DataFrames
df_opt_ctrl = pd.read_csv(filepath1)
df_brute_force = pd.read_csv(filepath2)
df_max_ac = pd.read_csv(filepath3)

# Extract means and standard errors
strategies = ["optimal control", "brute force", "max AC"]
means = [df_opt_ctrl["mean"][0], df_brute_force["mean"][0], df_max_ac["mean"][0]]
std_errors = [df_opt_ctrl["std_error"][0], df_brute_force["std_error"][0], df_max_ac["std_error"][0]]

# Create bar plot
x = np.arange(len(strategies))  # X-axis positions
plt.figure(figsize=(8, 8))
plt.style.use('fivethirtyeight')
bars = plt.bar(x, means, color=["green", "orange", "purple"], alpha=0.7)

# Adding error bars manually with thicker caps
plt.errorbar(x, means, yerr=std_errors, fmt='none', capsize=10, capthick=2, color="black")
# Labels and title
plt.xticks(x, strategies, fontsize=25, rotation=90)
plt.ylabel("Mean Well-Being Change")
plt.title("Comparison of Control Strategies")
plt.grid(axis="y", linestyle="--", alpha=0.6)

save_path =  os.path.join("results_control_strategies", "results_wellbeing_change.png")
plt.tight_layout()
plt.savefig(save_path, dpi=300)

# Show plot
plt.show()
