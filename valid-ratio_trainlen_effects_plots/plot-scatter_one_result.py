import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import myplots

filename = "trainlen1-eigenvaluesA.csv"

filepath = os.path.join("results_trainlen", filename)
df = pd.read_csv(filepath)
# Extract values
ratios = df.iloc[:, 0]
dominant_eigs = df.iloc[:, 1]

# Reverse the data
ratios_reversed = ratios[::-1]
dominant_eigenvalues_reversed = dominant_eigs[::-1]

# Use my plotting functions
save_path = os.path.splitext(filepath)[0] + "_plot.png"
#save_path = None
myplots.myplot_scatter(ratios_reversed, dominant_eigenvalues_reversed, xlabel='Valid Ratio', ylabel='Eigenvalues A', legend_label=None, save_path=save_path)