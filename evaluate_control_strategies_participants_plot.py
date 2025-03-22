import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load CSV
filepath = os.path.join("results_control_strategies", "results_test.csv")
df_results = pd.read_csv(filepath)  # Replace with your actual filename
save_path = os.path.splitext(filepath)[0] + "_plot.png"

# Convert columns to numeric (if needed)
df_results[["optimal control", "brute force", "max ac"]] = df_results[["optimal control", "brute force", "max ac"]].apply(pd.to_numeric)

df_melted = df_results.melt(id_vars=["file"], 
                            value_vars=["optimal control", "brute force", "max ac"],
                            var_name="Control Strategy", 
                            value_name="Wellbeing Change")

# Compute mean and standard error
summary_stats = df_melted.groupby("Control Strategy")["Wellbeing Change"].agg(["mean", "sem"]).reset_index()

plt.figure(figsize=(10, 6))
plt.style.use('fivethirtyeight')
sns.violinplot(data=df_melted, x="Control Strategy", y="Wellbeing Change", inner="point")

# Overlay Mean and Std Error
for i, row in summary_stats.iterrows():
    mean = row["mean"]
    sem = row["sem"]
    plt.errorbar(i, mean, yerr=sem, fmt='o', color='white', capsize=5, label="Mean ± SEM" if i == 0 else "")

plt.title("Distribution of Wellbeing Change Across Control Strategies")
plt.xlabel("")
plt.ylabel("Wellbeing Change")
plt.savefig(save_path, dpi=300)
plt.show()
