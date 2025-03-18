import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import myplots

folder = "results_ratio_trainlen_compare"

filenameRNN = "ratio_trainlen_mae_RNN_n10contin_mean.csv"
filepathRNN = os.path.join(folder, filenameRNN)
dfRNN = pd.read_csv(filepathRNN)
# Extract values
ratiosRNN = dfRNN["ratio"]
trainlensRNN = dfRNN["trainlen"]
errorsRNN = dfRNN["errors"]

filenameLDS = "ratio_trainlen_mae_LDS_n10contin_mean.csv"
filepathLDS = os.path.join(folder, filenameLDS)
dfLDS = pd.read_csv(filepathLDS)
# Extract values
ratiosLDS = dfLDS["ratio"]
trainlensLDS = dfLDS["trainlen"]
errorsLDS = dfLDS["errors"]

# Use my plotting function myplot_scatter_compare
x_ratios_lists = [ratiosRNN, ratiosLDS]
x_trainlen_lists = [trainlensRNN, trainlensLDS]
y_error_lists = [errorsRNN, errorsLDS]
colors = ['orangered', 'deepskyblue']
colors_rm = ['red','blue']
xlabel_ratio='Valid Ratio'
ylabel='MAE'
title='Compare RNN to LDS'
window_size = 100 # For running mean
markersize = 5
alpha = 0.5

legend_labels = ['Prediction Error RNN', 'Prediction Error LDS']

save_path = os.path.join(folder, "compare_ratio-mae_n10contin_mean_plot.png")
#save_path = None
# Plot valid ratio vs. prediction error
#myplots.myplot_scatter_compare(x_ratios_lists, y_error_lists, colors, colors_rm, window_size=window_size, markersize=markersize, alpha=alpha, xlabel=xlabel_ratio, ylabel=ylabel, title=title, legend_labels=legend_labels, save_path=save_path)

# Plot training set length vs. prediction error
xlabel_trainlen='Training Set Length'
save_path = os.path.join(folder, "compare_trainlen-mae_n10contin_mean_plot.png")
#save_path = None
myplots.myplot_scatter_compare(x_trainlen_lists, y_error_lists, colors,colors_rm, window_size=window_size, markersize=markersize, alpha=alpha, xlabel=xlabel_trainlen, ylabel=ylabel, title=title, legend_labels=legend_labels, save_path=save_path)
