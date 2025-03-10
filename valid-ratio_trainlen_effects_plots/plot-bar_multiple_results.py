import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import myplots


folder = "results_ratio"

filename1 = "trainratio-mae_True1.csv"
filepath1 = os.path.join(folder, filename1)
df1 = pd.read_csv(filepath1)
# Extract values
ratios = df1.iloc[:, 0]
y_values1 = df1.iloc[:, 1]
std_devs1 = df1.iloc[:, 2]
# Reverse the data (to plot from 20% to 80%)
ratios_reversed = ratios[::-1]
y_values_reversed1 = y_values1[::-1]
std_devs_reversed1 = std_devs1[::-1]

filename2 = "trainratio-mae_True10.csv"
filepath2 = os.path.join(folder, filename2)
df2 = pd.read_csv(filepath2)
# Extract values
y_values2 = df2.iloc[:, 1]
std_devs2 = df2.iloc[:, 2]
# Reverse the data (to plot from 20% to 80%)
y_values_reversed2 = y_values2[::-1]
std_devs_reversed2 = std_devs2[::-1]

# Use my plotting functions

x_lists = [ratios_reversed, ratios_reversed]
y_lists = [y_values_reversed1, y_values_reversed2]
yerr_lists = [std_devs_reversed1, std_devs_reversed2]
colors = ['b', 'g']
xlabel='Valid Ratio'
ylabel='MAE'
title='Compare different numbers of subsampling'

legend_labels = ['subsampling 1 time', 'subsampling 10 times']

#save_path = os.path.join(folder, "combined_plot.png")
save_path = None

# Plotting multiple lines on the same graph
myplots.myplot_bar_multiple(x_lists, y_lists, yerr_lists, colors, xlabel=xlabel, ylabel=ylabel, title=title, legend_labels=legend_labels, save_path=save_path)