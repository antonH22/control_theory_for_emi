import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import myplots

folder = "results_ratio_trainlen_compare"

filenameRNN = "ratio_trainlen_mae_RNN.csv"
filepathRNN = os.path.join(folder, filenameRNN)
dfRNN = pd.read_csv(filepathRNN)
# Extract values
ratiosRNN = dfRNN["ratio"]
trainlensRNN = dfRNN["trainlen"]
errorsRNN = dfRNN["errors"]

filenameLDS = "ratio_trainlen_mae_LDS.csv"
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
xlabel_ratio='Valid Ratio'
ylabel='MAE'
title='Compare RNN to LDS'

legend_labels = ['Prediction Error RNN', 'Prediction Error LDS']

#save_path = os.path.join(folder, "combined_compare_ratio-mae_plot.png")
save_path = None

# Plot valid ratio vs. prediction error
myplots.myplot_scatter_compare(x_ratios_lists, y_error_lists, colors, xlabel=xlabel_ratio, ylabel=ylabel, title=title, legend_labels=legend_labels, save_path=save_path)

# Plot training set length vs. prediction error
xlabel_ratio='Training Set Length'
myplots.myplot_scatter_compare(x_trainlen_lists, y_error_lists, colors, xlabel=xlabel_ratio, ylabel=ylabel, title=title, legend_labels=legend_labels, save_path=save_path)
