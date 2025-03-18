import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import myplots


folder = "results_ratio"

filename = 'completeratio-everything1_locf_reduction.csv'
filepath = os.path.join(folder, filename)

df = pd.read_csv(filepath)
# Extract values
results_ratios = df["ratio"]
results_eigenvalues_A = df["eigenvalues"]
results_frobenius_A = df["frobenius_A"]
results_frobenius_K = df["frobenius_K"]

print(len(results_eigenvalues_A))

# Use my plotting function myplot_scatter_compare
x_lists = [results_ratios, results_ratios, results_ratios]
y_lists = [results_eigenvalues_A, results_frobenius_A, results_frobenius_K]
colors = ['orangered', 'deepskyblue','yellowgreen']
colors_rm = ['red', 'blue', 'green']
xlabel_ratio='Valid Ratio'
title='Eigenvalue A and frobenius norm A+K for each dataset'

legend_labels = ['Eigenvalues A', 'Frobenius_A', 'Frobenius_K']

save_path = os.path.join(folder, "completeratio-scatter1_locf_reduction_plot.png")
#save_path = None
# Plot valid ratio vs. prediction error
window_size = 100
markersize = 30
myplots.myplot_scatter_compare(x_lists, y_lists, colors, colors_rm,window_size=window_size, markersize=markersize, xlabel=xlabel_ratio, title=title, legend_labels=legend_labels, save_path=save_path)