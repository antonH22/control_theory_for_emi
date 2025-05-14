import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import myplots

### Plot comparison of RNN and LDS prediction accuracy across empirical valid ratios and training set lengths (5.1.5, Figure 6)

df = pd.read_csv(os.path.join("results_replicated", "mae_vs_ratio_trainlen.csv"))
# Extract values
ratios = df["ratio"]
trainlens = df["trainlen"]
mae_rnn = df["mae_rnn"]
mae_lds = df["mae_lds"]

# Use my plotting function myplot_scatter_compare
x_ratios_lists = [ratios, ratios]
x_trainlen_lists = [trainlens, trainlens]
y_error_lists = [mae_lds, mae_rnn]
colors = ['blue', 'indianred']
colors_rm = ['blue', 'indianred']
xlabel_ratio='Valid ratio'
xlabel_trainlen='Training set length'
ylabel='MAE'
title=None
window_size = 100 # For running mean
markersize = 30
alpha = 0.5

legend_labels = ['LDS', 'RNN']

rm_only = False
save_path = None
# Plot valid ratio vs. prediction error
myplots.myplot_scatter_compare(x_ratios_lists, y_error_lists, colors, colors_rm, rm_only=rm_only, window_size=window_size, markersize=markersize, alpha=alpha, xlabel=xlabel_ratio, ylabel=ylabel, title=title, legend_labels=legend_labels, save_path=save_path)

# Plot training set length vs. prediction error
save_path = None
myplots.myplot_scatter_compare(x_trainlen_lists, y_error_lists, colors,colors_rm, rm_only=rm_only, window_size=window_size, markersize=markersize, alpha=alpha, xlabel=xlabel_trainlen, ylabel=ylabel, title=title, legend_labels=legend_labels, save_path=save_path)

rm_only = True
save_path = None
# Plot valid ratio vs. prediction error
myplots.myplot_scatter_compare(x_ratios_lists, y_error_lists, colors, colors_rm, rm_only=rm_only, window_size=window_size, markersize=markersize, alpha=alpha, xlabel=xlabel_ratio, ylabel=ylabel, title=title, legend_labels=legend_labels, save_path=save_path)

# Plot training set length vs. prediction error
save_path = None
myplots.myplot_scatter_compare(x_trainlen_lists, y_error_lists, colors,colors_rm, rm_only=rm_only, window_size=window_size, markersize=markersize, alpha=alpha, xlabel=xlabel_trainlen, ylabel=ylabel, title=title, legend_labels=legend_labels, save_path=save_path)
