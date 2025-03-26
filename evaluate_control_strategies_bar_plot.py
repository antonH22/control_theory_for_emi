import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load CSV
filepath = os.path.join("results_control_strategies", "results_rnn_steps_mean.csv")
df = pd.read_csv(filepath)  # Replace with your actual filename
save_path = os.path.splitext(filepath)[0] + "_plot.png"

# Extract the prediction steps (1-12)
n_steps = [1,2,3,4,5,6,7,8,9,10,11,12]
strategies = ["optimal control", "brute force", "max ac"]


colors = {
    "optimal control": "green",
    "brute force": "orange",
    "max ac": "purple"
}

# Compute means and standard deviations
means = {strategy: [] for strategy in strategies}
stdevs = {strategy: [] for strategy in strategies}

for step in n_steps:
    for strategy in strategies:
        column_name = f"{strategy} (n={step})"
        if column_name in df.columns:
            means[strategy].append(df[column_name].mean())
            stdevs[strategy].append(df[column_name].std())
plt.style.use('fivethirtyeight')
# Plot
fig, ax = plt.subplots(figsize=(15, 6))
bar_width = 0.25
x_indexes = np.arange(len(n_steps))

# Plot bars with error bars
for i, strategy in enumerate(strategies):
    ax.bar(x_indexes + i * bar_width, means[strategy], yerr=stdevs[strategy], 
           capsize=3, error_kw={'elinewidth': 1, 'capthick': 1}, label=strategy, width=bar_width, color=colors[strategy], alpha=0.7)

# Formatting
ax.grid(axis='x')  # Only keep horizontal grid lines
ax.set_xticks(x_indexes + bar_width)
ax.set_xticklabels(n_steps)
ax.set_xlabel("Prediction Length (n_steps)")
ax.set_ylabel("Wellbeing Change")
ax.set_title("Comparison of Control Strategies")
ax.legend()
plt.tight_layout()
plt.savefig(save_path)
plt.show()



