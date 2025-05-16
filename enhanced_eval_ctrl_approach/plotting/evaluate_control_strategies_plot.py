import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

### Plot for evaluating control strategies with bias correction (5.1.6, Figure 7) and without bias correction (A.2, Figure 10)

# Load CSV
filepath = os.path.join("results_replicated", "results_control_strategies_offline_bias_corrected.csv")
df = pd.read_csv(filepath)

n_steps = [1,2,3,4,5,6,7,8,9,10,11,12]
strategies = ["optimal control", "brute force", "max ac"] # Optional "real emi", "no emi", "bias" (adjust bar_width)

colors = {
    "optimal control": "green",
    "brute force": "orange",
    "max ac": "purple",
    "real emi" : "grey",
    "no emi" : "black",
    "bias": "red"
}

# Compute means and standard deviations
means = {strategy: [] for strategy in strategies}
stdevs = {strategy: [] for strategy in strategies}
sterrs = {strategy: [] for strategy in strategies}

# Average the results over all participants per step
for strategy in strategies:
    for step in n_steps:
        column_name = f"{strategy} (n={step})"
        means[strategy].append(df[column_name].mean())
        stdevs[strategy].append(df[column_name].std())
        sterrs[strategy].append(df[column_name].sem())

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

fontsize = 20
ticksize = 18
# Formatting
ax.grid(axis='x')  # Only keep horizontal grid lines
ax.set_xticks(x_indexes + bar_width)
ax.set_xticklabels(n_steps)
ax.tick_params(axis='both', which='major', labelsize=ticksize)
ax.set_xlabel(r"$r$-th step", fontsize=fontsize)
ax.set_ylabel(r"$\Delta \mathcal{W}$", fontsize=fontsize)
ax.legend(loc='upper right', fontsize=fontsize)
plt.ylim(None, 0.176)
plt.ylim(None, 0.21)
plt.tight_layout()
plt.show()