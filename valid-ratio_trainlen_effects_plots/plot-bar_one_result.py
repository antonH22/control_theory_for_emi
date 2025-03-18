import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import myplots

filename = "trainlen_frobenius_AC.csv"

filepath = os.path.join("results_trainlen", filename)
df = pd.read_csv(filepath)
# Extract values
ratios = df.iloc[:, 0]
y_values = df.iloc[:, 1]
std_errs = df["se"]
"""
num_elements = df.iloc[:, 3] 

print(f'std dev mean: {np.mean(std_devs)}')
for i,_ in enumerate(ratios):
    print(f'Number of values for {ratios[i]}: {num_elements[i]}')"
"""

# Reverse the data (to plot from 20% to 80%)
ratios_reversed = ratios[::-1]
y_values_reversed = y_values[::-1]
std_errs_reversed = std_errs[::-1]

# Use my plotting functions
save_path = os.path.splitext(filepath)[0] + "_logplot.png"
#save_path = None
legend_label = 'L2 Norm AC'
xlabel='Training Set Length'
ylabel='L2 Norm AC'
title=f'{xlabel} vs. {ylabel}'
log_scale = True
myplots.myplot_bar(ratios_reversed, y_values_reversed, std_errs_reversed, log_scale=log_scale, xlabel=xlabel, ylabel=ylabel, title=title, legend_label=legend_label, save_path=save_path)
