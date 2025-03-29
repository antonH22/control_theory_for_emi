import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load CSV
filepath = os.path.join("results_control_strategies", "results_rnn_offline_step.csv")
df = pd.read_csv(filepath)
save_path = os.path.splitext(filepath)[0] + "_acc_plot.png"

# Extract the prediction steps (1-12)
n_steps = [1,2,3,4,5,6,7,8,9,10,11,12]
strategies = ["optimal control", "brute force", "max ac", "no emi"]


colors = {
    "optimal control": "green",
    "brute force": "orange",
    "max ac": "purple",
    "no emi": "grey"
}

# Compute means and standard deviations
means = {strategy: [] for strategy in strategies}
stdevs = {strategy: [] for strategy in strategies}
sterrs = {strategy: [] for strategy in strategies}

for strategy in strategies:
    for step in n_steps:
        column_name = f"{strategy} (n={step})"
        means[strategy].append(df[column_name].mean())
        stdevs[strategy].append(df[column_name].std())
        sterrs[strategy].append(df[column_name].sem())

# Compute means and standard deviations over accumulated values over the steps for each strategy
acc_means = {strategy: [] for strategy in strategies}
acc_stdevs = {strategy: [] for strategy in strategies}
acc_sterrs = {strategy: [] for strategy in strategies}

for strategy in strategies:
    accumulated_values = []
    for step in n_steps:
        column_name = f"{strategy} (n={step})" 
        accumulated_values.extend(df[column_name])
        acc_means[strategy].append(pd.Series(accumulated_values).mean())
        acc_stdevs[strategy].append(pd.Series(accumulated_values).std())
        acc_sterrs[strategy].append(pd.Series(accumulated_values).sem())

plt.style.use('fivethirtyeight')
# Plot
fig, ax = plt.subplots(figsize=(15, 6))
bar_width = 0.2
x_indexes = np.arange(len(n_steps))

# Downscale standard deviation
divided_stdevs = {strategy: np.array(stdevs[strategy]) / 2 for strategy in strategies}

# Plot bars with error bars
for i, strategy in enumerate(strategies):
    ax.bar(x_indexes + i * bar_width, means[strategy], yerr=sterrs[strategy], 
           capsize=3, error_kw={'elinewidth': 1, 'capthick': 1}, label=strategy, width=bar_width, color=colors[strategy], alpha=0.7)

# Formatting
ax.grid(axis='x')  # Only keep horizontal grid lines
ax.set_xticks(x_indexes + bar_width)
ax.set_xticklabels(n_steps)
ax.set_xlabel("Difference at the n-th step after intervention between real and predicted data")
ax.set_ylabel("Wellbeing Change")
ax.set_title("Comparison of Control Strategies (offline, accumulated)")
ax.legend(loc='upper right')
plt.tight_layout()
#plt.savefig(save_path)
plt.show()



