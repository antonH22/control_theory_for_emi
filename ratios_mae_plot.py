import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

filename = "results_test.csv"
filepath = os.path.join("results_ratio-mae", filename)
df = pd.read_csv(filepath)
# Extract values
ratios = df["ratio"]
mean_errors = df["mean_error"]
std_errors = df["std_dev"]

print(f'std_errors mean: {np.mean(std_errors)}')

# Reverse the data (to plot from 20% to 80%)
ratios_reversed = ratios[::-1]
mean_errors_reversed = mean_errors[::-1]
std_errors_reversed = std_errors[::-1]

# Create the plot
plt.figure(figsize=(8, 6))
plt.errorbar(ratios_reversed, mean_errors_reversed, yerr=std_errors_reversed, fmt='-o', capsize=5, label='Mean Error ± Std Dev')

# Customize the plot
plt.title('Mean Error vs Ratio with Standard Deviation')
plt.xlabel('Ratios')
plt.ylabel('Mean Error')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()