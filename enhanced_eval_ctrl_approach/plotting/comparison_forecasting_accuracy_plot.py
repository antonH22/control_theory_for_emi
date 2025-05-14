import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import myplots

### Plot comparison of RNN and LDS forecasting performance (5.1.5, Figure 5)

n_steps = list(range(1, 13))

# Load CSV files
filename1 = "mae_lds_forecast.csv"
filepath1 = os.path.join("results_replicated", filename1)
df_lds = pd.read_csv(filepath1)

filename2 = "mae_rnn_forecast.csv"
filepath2 = os.path.join("results_replicated", filename2)
df_rnn = pd.read_csv(filepath2)

# Compute column-wise mean and standard error
means_lds = df_lds.mean()
stderr_lds = df_lds.sem()

means_rnn = df_rnn.mean()
stderr_rnn = df_rnn.sem()

x_values = [n_steps, n_steps]
y_values = [means_lds, means_rnn]
yerrs = [stderr_lds, stderr_rnn]
colors = ["blue", "indianred"]

# Use my plotting functions
save_path = None
legend_label = None
xlabel="Forecasting Step (r)"
ylabel='MAE'
legend_labels = ["LDS", "RNN"]
myplots.myplot_bar_multiple(x_values, y_values, yerr_lists=yerrs, xlabel=xlabel, ylabel=ylabel, colors=colors, legend_labels=legend_labels, save_path=save_path)