import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define directory where the files are stored
directory = "results_ratio-mae"

# List of filenames
filenames = [
    "trainratio-mae1_we.csv",
    "trainratio-mae10_we.csv",
    "trainratio-mae50_we.csv"
]

# Function to read the file and return the relevant columns
def read_file(filename):
    filepath = os.path.join(directory, filename)
    df = pd.read_csv(filepath)
    ratios = df.iloc[:, 0]
    y_values = df.iloc[:, 1]
    std_devs = df.iloc[:, 2]
    num_elements = df.iloc[:, 3]
    return ratios, y_values, std_devs, num_elements

# Create the plot
plt.figure(figsize=(8, 6))

# Iterate over filenames, read the data, reverse it, and plot
for filename in filenames:
    ratios, y_values, std_devs, num_elements = read_file(filename)
    
    # Reverse the data (to plot from 20% to 80%)
    ratios_reversed = ratios[::-1]
    y_values_reversed = y_values[::-1]
    std_devs_reversed = std_devs[::-1]
    
    # Plot data from each file
    plt.errorbar(ratios_reversed, y_values_reversed, yerr=std_devs_reversed, fmt='-o', capsize=5, label=filename)

# Customize the plot
plt.title('Valid Ratio Effect: Comparison of different number of resampling')
plt.xlabel('Ratios')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()