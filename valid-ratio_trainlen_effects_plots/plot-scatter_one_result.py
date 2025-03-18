import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import myplots

filename = "trainlen-eigenvaluesA_len160.csv"

filepath = os.path.join("results_trainlen", filename)
df = pd.read_csv(filepath)
# Extract values
ratios = df.iloc[:, 0]
dominant_eigs = df.iloc[:, 1]

# Reverse the data
ratios_reversed = ratios[::-1]
dominant_eigenvalues_reversed = dominant_eigs[::-1]

# Use my plotting functions
save_path = os.path.splitext(filepath)[0] + "plot.png"
#save_path = None
xlabel='Training Set Length'
ylabel='Dominant Eigenvalue A'
legend_label=None
title=f'{xlabel} vs. {ylabel}'
alpha = 1.0

myplots.myplot_scatter(ratios_reversed, dominant_eigenvalues_reversed, alpha=alpha, xlabel=xlabel, ylabel=ylabel, legend_label=legend_label, save_path=save_path)