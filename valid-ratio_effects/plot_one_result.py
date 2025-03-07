import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filename = "trainratio-frobeniusA1.csv"

filepath = os.path.join("results_ratio", filename)
df = pd.read_csv(filepath)
# Extract values
ratios = df.iloc[:, 0]
y_values = df.iloc[:, 1]
std_devs = df.iloc[:, 2]
num_elements = df.iloc[:, 3] 

print(f'std dev mean: {np.mean(std_devs)}')
for i,_ in enumerate(ratios):
    print(f'Number of values for {ratios[i]}: {num_elements[i]}')

# Reverse the data (to plot from 20% to 80%)
ratios_reversed = ratios[::-1]
y_values_reversed = y_values[::-1]
std_devs_reversed = std_devs[::-1]

# Create the plot
plt.figure(figsize=(8, 6))
plt.errorbar(ratios_reversed, y_values_reversed, yerr=std_devs_reversed, fmt='-o', capsize=5, label='Mean Error ± Std Dev')

# Customize the plot
plt.title(f'Valid Ratio effect: {filename}')
plt.xlabel('Ratios')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()