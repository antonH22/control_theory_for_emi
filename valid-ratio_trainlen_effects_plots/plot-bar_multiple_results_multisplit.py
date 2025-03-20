import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import myplots


folder = "results_ratio"

filename = "results_30ratio80_splits4.csv"

filepath = os.path.join("results_ratio", filename)
# Read CSV file
df = pd.read_csv(filepath)
ratio_list = df["Ratio"].unique().tolist()
# Create lists to hold x (ratios) and y (MAE values) for each participant
x_lists = [ratio_list, ratio_list, ratio_list]
y_lists = []
yerr_lists = []

# Compute mean and std deviation per ratio across all participants
mean_frobA_per_ratio = df.groupby("Ratio")["Frobenius A"].mean().to_list()
sem_frobA_per_ratio = df.groupby("Ratio")["Frobenius A"].sem().to_list()
y_lists.append(mean_frobA_per_ratio[::-1])
yerr_lists.append(sem_frobA_per_ratio[::-1])

mean_frobK_per_ratio = df.groupby("Ratio")["Frobenius K"].mean().to_list()
sem_frobK_per_ratio = df.groupby("Ratio")["Frobenius K"].sem().to_list()
y_lists.append(mean_frobK_per_ratio[::-1])
yerr_lists.append(sem_frobK_per_ratio[::-1])

mean_mae_per_ratio = df.groupby("Ratio")["Mean MAE"].mean().to_list()
sem_mae_per_ratio = df.groupby("Ratio")["Mean MAE"].sem().to_list()
y_lists.append(mean_mae_per_ratio[::-1])
yerr_lists.append(sem_mae_per_ratio[::-1])

# Use my plotting function
colors = ["blue", "dodgerblue", "dimgray"]
xlabel='Valid Ratio'
ylabel=''
title='Compare MAE and Frobenius Norms'
log_scale=True

legend_labels = ['Frobenius A','Frobenius K','MAE']

save_path = os.path.join(folder, filename[:-4]+"compare_plot.png")
#save_path = None

# Plotting multiple lines on the same graph
myplots.myplot_bar_multiple(x_lists, y_lists, yerr_lists, colors, xlabel=xlabel, ylabel=ylabel, title=title, legend_labels=legend_labels,log_scale=log_scale, save_path=save_path)