import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

filename = "trainlen-mae1.csv"
filepath = os.path.join("results_trainlen-mae", filename)
df = pd.read_csv(filepath)
# Extract values
train_set_lengths = df["train_set_length"]
mean_errors = df["mean_error"]
std_errors = df["std_dev"]

print(f'std_errors mean: {np.mean(std_errors)}')

# Reverse the data (to plot from 20% to 80%)
len_reversed = train_set_lengths[::-1]
mean_errors_reversed = mean_errors[::-1]
std_errors_reversed = std_errors[::-1]

# Create the plot
plt.figure(figsize=(8, 6))
plt.errorbar(len_reversed,  mean_errors_reversed, yerr=std_errors_reversed, fmt='-o', capsize=5, label='Mean Error ± Std Dev')

# Customize the plot
plt.title('Mean Error vs Train Set Length with Standard Deviation')
plt.xlabel('Train Set Length')
plt.ylabel('Mean Error')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()