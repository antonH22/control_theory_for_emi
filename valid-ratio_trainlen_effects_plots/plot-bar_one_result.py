import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import myplots

filename = "trainratio-mae_True10.csv"

filepath = os.path.join("results_ratio", filename)
df = pd.read_csv(filepath)
# Extract values
ratios = df.iloc[:, 0]
y_values = df.iloc[:, 1]
std_devs = df.iloc[:, 2]
"""
num_elements = df.iloc[:, 3] 

print(f'std dev mean: {np.mean(std_devs)}')
for i,_ in enumerate(ratios):
    print(f'Number of values for {ratios[i]}: {num_elements[i]}')"
"""

# Reverse the data (to plot from 20% to 80%)
ratios_reversed = ratios[::-1]
y_values_reversed = y_values[::-1]
std_devs_reversed = std_devs[::-1]

# Use my plotting functions
#save_path = os.path.splitext(filepath)[0] + "_plot2.png"
save_path = None
myplots.myplot_bar(ratios_reversed, y_values_reversed, std_devs_reversed, xlabel='Train Set Length', ylabel='Frobenius K', legend_label=None, save_path=save_path)