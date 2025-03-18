import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import myplots


folder = "results_ratio"

filename = "mae_True10_mean.csv"
filepath = os.path.join(folder, filename)
df = pd.read_csv(filepath)
# Extract values
ratios = df.iloc[:, 0]
y_values = df.iloc[:, 1]
std_errs = df["std_error"]

ratios_reversed = ratios[::-1]
y_values_reversed = y_values[::-1]
std_errs_reversed = std_errs[::-1]


filename1 = "frobeniusA10.csv"
filepath1 = os.path.join(folder, filename1)
df1 = pd.read_csv(filepath1)
# Extract values
y_values1 = df1.iloc[:, 1]
std_errs1 = df1['se']
# Reverse the data (to plot from 20% to 80%)
y_values_reversed1 = y_values1[::-1]
std_errs_reversed1 = std_errs1[::-1]


filename2 = "frobeniusK10.csv"
filepath2 = os.path.join(folder, filename2)
df2 = pd.read_csv(filepath2)
# Extract values
y_values2 = df2.iloc[:, 1]
std_errs2 = df2['se']
# Reverse the data (to plot from 20% to 80%)
y_values_reversed2 = y_values2[::-1]
std_errs_reversed2 = std_errs2[::-1]

# Use my plotting function
x_lists = [ratios_reversed,ratios_reversed, ratios_reversed]
y_lists = [y_values_reversed,y_values_reversed1, y_values_reversed2]
yerr_lists = [std_errs_reversed,std_errs_reversed1, std_errs_reversed2]
colors = ['blue','brown','g']
xlabel='Valid Ratio'
ylabel=''
title='Compare MAE and Frobenius Norms'
log_scale=False

legend_labels = ['MAE','Frobenius A', 'Frobenius K']

save_path = os.path.join(folder, "frobenius_A_K_MAE_plot.png")
#save_path = None

# Plotting multiple lines on the same graph
myplots.myplot_bar_multiple(x_lists, y_lists, yerr_lists, colors, xlabel=xlabel, ylabel=ylabel, title=title, legend_labels=legend_labels,log_scale=log_scale, save_path=save_path)