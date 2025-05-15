import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import myplots
import numpy as np

### Plot for evaluating control strategies across valid ratios and training set length (A.3, Figure 11, 12)

def summarize_local_averages(x, y, x_targets, radius=0.05):
    x_avg = []
    y_avg = []
    y_se = []
    counts = []

    x = np.asarray(x)
    y = np.asarray(y)

    for target in x_targets:
        mask = (x >= target - radius) & (x <= target + radius)
        if np.any(mask):
            y_vals = y[mask]
            x_avg.append(target)
            y_avg.append(np.mean(y_vals))
            y_se.append(np.std(y_vals, ddof=1) / np.sqrt(len(y_vals)))
            counts.append(len(y_vals))
    return (np.array(x_avg), np.array(y_avg), np.array(y_se), np.array(counts))

step = 1
folder = "results_replicated"

filename = "results_control_strategies_online_ratio_trainlen.csv"
#filename = "results_control_strategies_onlineratio_trainlen.csv"
filepath = os.path.join(folder, filename)
df = pd.read_csv(filepath)
# Extract values
ratios = df["valid ratio"]
trainlens = df["trainset length"]
brute_force_wellbeing_difference = df["optimal control"]

print(df)

color = "green"
xlabel_ratio='Valid ratio'
xlabel_trainlen='Training set length'
ylabel=r"$\Delta \mathcal{W}$"
title=f'Relation of valid ratio to brute force online results step {step}'
title=None

window_size = None
markersize = 30
alpha = 0.5
legend_label = None

rm_only = False


save_path = None
myplots.myplot_scatter(ratios, brute_force_wellbeing_difference, rm_only=rm_only, color_sc=color, color_rm="darkorange", window_size=window_size, markersize=markersize, alpha=alpha, legend_label=legend_label, xlabel=xlabel_ratio, ylabel=ylabel, title=title, save_path=save_path)

myplots.myplot_scatter(trainlens, brute_force_wellbeing_difference, rm_only=rm_only, color_sc=color, color_rm="darkorange", window_size=window_size, markersize=markersize, alpha=alpha, xlabel=xlabel_trainlen, ylabel=ylabel, title=title, save_path=save_path)


window_ratio = 0.1
x_targets = np.arange(0.2, 1.01, window_ratio)
x_avg, y_avg, y_se, counts = summarize_local_averages(ratios, brute_force_wellbeing_difference, x_targets, radius=window_ratio/2)
myplots.myplot_bar(x_avg, y_avg, y_se, counts=counts, color="green", save_path=save_path, xlabel=xlabel_ratio, ylabel=ylabel)

window_trainlen = 20
x_targets_trainlen = np.arange(80, 240, window_trainlen)
x_avg, y_avg, y_se, counts = summarize_local_averages(trainlens, brute_force_wellbeing_difference, x_targets_trainlen, radius=window_trainlen/2)
myplots.myplot_bar(x_avg, y_avg, y_se, counts=counts, color="green", save_path=save_path, xlabel=xlabel_trainlen, ylabel=ylabel)