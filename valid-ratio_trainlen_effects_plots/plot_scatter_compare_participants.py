import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import myplots
import matplotlib.pyplot as plt


filename = "mae_10ratio80_splits4_scatter.csv"

filepath = os.path.join("results_ratio", filename)
# Read CSV file
df = pd.read_csv(filepath)

# Extract unique participants (files)
participants = df["File"].unique()

# Create lists to hold x (ratios) and y (MAE values) for each participant
x_lists = []
y_lists = []
legend_labels = []
colors = plt.cm.get_cmap('tab10', len(participants))  # Assign different colors

# Organize data per participant
for i, participant in enumerate(participants):
    df_participant = df[df["File"] == participant]  # Filter data for one participant
    x_lists.append(df_participant["Ratio"].tolist())
    y_lists.append(df_participant["Mean MAE"].tolist())
    legend_labels.append(participant.split("\\")[-1])  # Extract filename for legend

# Compute mean and std deviation per ratio across all participants
mean_mae_per_ratio = df.groupby("Ratio")["Mean MAE"].mean().to_list()
std_mae_per_ratio = df.groupby("Ratio")["Mean MAE"].sem().to_list()

# Call the plotting function
myplots.myplot_scatter_compare_participants(x_lists, y_lists, colors=[colors(i) for i in range(len(participants))],
                        mean=mean_mae_per_ratio, std_error=std_mae_per_ratio,
                       xlabel="Ratio", ylabel="Mean MAE", title="Compare datasets with high valid ratio", 
                       legend_labels=legend_labels, save_path = os.path.splitext(filepath)[0] + "_plot.png")