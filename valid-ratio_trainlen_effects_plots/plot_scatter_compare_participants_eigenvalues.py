import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import myplots
import matplotlib.pyplot as plt
import ast


filename = "eigenvaluesA1ratio80_participants_testfilenames.csv"

filepath = os.path.join("results_ratio", filename)
# Read CSV file
df = pd.read_csv(filepath)

# Convert the columns from a string to an actual list
df["ratios"] = df["ratio"].apply(ast.literal_eval)
df["eigenvalues"] = df["eigenvalues"].apply(ast.literal_eval)

x_lists = df["ratios"].tolist()
y_lists = df["eigenvalues"].tolist()
filenames = df["filenames"].tolist()

colors = plt.cm.get_cmap('tab20', len(x_lists))  # Use a colormap for different participants

# Call the plotting function
myplots.myplot_scatter_compare_participants(x_lists, y_lists, colors=[colors(i) for i in range(len(x_lists))],
                       xlabel="Ratio", ylabel="Eigenvalue", title="Compare datasets with high valid ratio", 
                       legend_labels=filenames, save_path = os.path.splitext(filepath)[0] + "_plot.png")