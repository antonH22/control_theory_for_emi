import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import myplots
import matplotlib.pyplot as plt


filename = "results_30ratio80_splits4_new.csv"

filepath = os.path.join("results_ratio", filename)
# Read CSV file
df = pd.read_csv(filepath)

# Extract unique participants (files)
participants = df["File"].unique()

# Create lists to hold x (ratios) and y (MAE values) for each participant
x_lists = []
y_lists_mae = []
y_lists_eig = []
y_lists_frobA = []
y_lists_frobK = []
y_lists_frobAC = []
y_lists_meanA = []
y_lists_varA = []
y_lists_corrA = []
legend_labels = []
colors = plt.cm.get_cmap('tab10', len(participants))  # Assign different colors

# Organize data per participant (mean over different splits)
for i, participant in enumerate(participants):
    df_participant = df[df["File"] == participant]  # Filter data for one participant
    x_lists.append(df_participant["Ratio"].tolist())
    y_lists_mae.append(df_participant["Mean MAE"].tolist())
    y_lists_eig.append(df_participant["Mean Eigenvalue"].tolist())
    y_lists_frobA.append(df_participant["Frobenius A"].tolist())
    y_lists_frobK.append(df_participant["Frobenius K"].tolist())
    y_lists_frobAC.append(df_participant["Frobenius AC"].tolist())
    y_lists_meanA.append(df_participant["Variance A"].tolist())
    y_lists_varA.append(df_participant["Variance A"].tolist())
    y_lists_corrA.append(df_participant["Correlation A"].tolist())

    legend_labels.append(participant.split("\\")[-1])  # Extract filename for legend

# Compute mean and std deviation per ratio across all participants
mean_mae_per_ratio = df.groupby("Ratio")["Mean MAE"].mean().to_list()
std_mae_per_ratio = df.groupby("Ratio")["Mean MAE"].sem().to_list()

mean_eig_per_ratio = df.groupby("Ratio")["Mean Eigenvalue"].mean().to_list()
std_eig_per_ratio = df.groupby("Ratio")["Mean Eigenvalue"].sem().to_list()

mean_frobA_per_ratio = df.groupby("Ratio")["Frobenius A"].mean().to_list()
std_frobA_per_ratio = df.groupby("Ratio")["Frobenius A"].sem().to_list()

mean_frobK_per_ratio = df.groupby("Ratio")["Frobenius K"].mean().to_list()
std_frobK_per_ratio = df.groupby("Ratio")["Frobenius K"].sem().to_list()

#mean_frobAC_per_ratio = df.groupby("Ratio")["Frobenius AC"].mean().to_list()
#std_frobAC_per_ratio = df.groupby("Ratio")["Frobenius AC"].sem().to_list()
# Compute median per ratio
median_frobAC_per_ratio = df.groupby("Ratio")["Frobenius AC"].median().to_list()
"""
# Compute IQR per ratio
q1_frobAC_per_ratio = df.groupby("Ratio")["Frobenius AC"].quantile(0.25).to_list()
q3_frobAC_per_ratio = df.groupby("Ratio")["Frobenius AC"].quantile(0.75).to_list()
iqr_frobAC_per_ratio = [q3 - q1 for q1, q3 in zip(q1_frobAC_per_ratio, q3_frobAC_per_ratio)]
# Compute lower and upper errors for error bars
lower_error = [median - q1 for median, q1 in zip(median_frobAC_per_ratio, q1_frobAC_per_ratio)]
upper_error = [q3 - median for median, q3 in zip(median_frobAC_per_ratio, q3_frobAC_per_ratio)]
yerr_iqr = [lower_error, upper_error]  # Asymmetric error format
"""

mean_meanA_per_ratio = df.groupby("Ratio")["Mean A"].mean().to_list()
std_meanA_per_ratio = df.groupby("Ratio")["Mean A"].sem().to_list()

mean_varA_per_ratio = df.groupby("Ratio")["Variance A"].mean().to_list()
std_varA_per_ratio = df.groupby("Ratio")["Variance A"].sem().to_list()

mean_corrA_per_ratio = df.groupby("Ratio")["Correlation A"].mean().to_list()
std_corrA_per_ratio = df.groupby("Ratio")["Correlation A"].sem().to_list()


# Call the plotting function for MAE
myplots.myplot_scatter_compare_participants(x_lists, y_lists_mae, colors=[colors(i) for i in range(len(participants))],
                        mean=mean_mae_per_ratio, std_error=std_mae_per_ratio,
                       xlabel="Ratio", ylabel="Mean MAE", title="Compare datasets with high valid ratio", 
                       legend_labels=legend_labels, save_path = os.path.splitext(filepath)[0] + "_mae_plot.png")

# Call the plotting function for Eigenvalue
myplots.myplot_scatter_compare_participants(x_lists, y_lists_eig, colors=[colors(i) for i in range(len(participants))],
                        mean=mean_eig_per_ratio, std_error=std_eig_per_ratio,
                       xlabel="Ratio", ylabel="Eigenvalue", title="Compare datasets with high valid ratio", 
                       legend_labels=legend_labels, save_path = os.path.splitext(filepath)[0] + "_eig_plot.png")

# Call the plotting function for Frobenius A
myplots.myplot_scatter_compare_participants(x_lists, y_lists_frobA, colors=[colors(i) for i in range(len(participants))],
                        mean=mean_frobA_per_ratio, std_error=std_frobA_per_ratio,
                       xlabel="Ratio", ylabel="Frobenius Norm A - I", title="Compare datasets with high valid ratio", 
                       legend_labels=legend_labels, save_path = os.path.splitext(filepath)[0] + "_frobA_diff_plot.png")

# Call the plotting function for Frobenius K
myplots.myplot_scatter_compare_participants(x_lists, y_lists_frobK, colors=[colors(i) for i in range(len(participants))],
                        mean=mean_frobK_per_ratio, std_error=std_frobK_per_ratio,
                       xlabel="Ratio", ylabel="Frobenius Norm K", title="Compare datasets with high valid ratio", 
                       legend_labels=legend_labels, save_path = os.path.splitext(filepath)[0] + "_frobK_plot.png")

# Call the plotting function for Frobenius AC
myplots.myplot_scatter_compare_participants(x_lists, y_lists_frobAC, colors=[colors(i) for i in range(len(participants))],
                        mean=median_frobAC_per_ratio, std_error=None, log_scale=True,
                       xlabel="Ratio", ylabel="Frobenius Norm AC", title="Compare datasets with high valid ratio", 
                       legend_labels=legend_labels, errorlabel="Median", save_path = os.path.splitext(filepath)[0] + "_frobAC_plot.png")

# Call the plotting function for Correlation A
myplots.myplot_scatter_compare_participants(x_lists, y_lists_corrA, colors=[colors(i) for i in range(len(participants))],
                        mean=mean_corrA_per_ratio, std_error=std_corrA_per_ratio, log_scale=False,
                       xlabel="Ratio", ylabel="Correlation", title="Correlation for each ratio to the matrix with ratio 0.8", 
                       legend_labels=legend_labels, save_path = os.path.splitext(filepath)[0] + "_corrA_plot.png")

# Call the plotting function for Mean A
myplots.myplot_scatter_compare_participants(x_lists, y_lists_meanA, colors=[colors(i) for i in range(len(participants))],
                        mean=mean_meanA_per_ratio, std_error=std_meanA_per_ratio, log_scale=False,
                       xlabel="Ratio", ylabel="Mean A", title="Mean of A per ratio", 
                       legend_labels=legend_labels, save_path = os.path.splitext(filepath)[0] + "_meanA_plot.png")

# Call the plotting function for Variance A
myplots.myplot_scatter_compare_participants(x_lists, y_lists_varA, colors=[colors(i) for i in range(len(participants))],
                        mean=mean_varA_per_ratio, std_error=std_varA_per_ratio, log_scale=False,
                       xlabel="Ratio", ylabel="Variance A", title="Variance of A per ratio", 
                       legend_labels=legend_labels, save_path = os.path.splitext(filepath)[0] + "_varA_plot.png")