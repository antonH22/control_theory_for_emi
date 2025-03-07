import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filename = "trainratio-eigenvaluesA1.csv"

filepath = os.path.join("results_ratio", filename)
df = pd.read_csv(filepath)
# Extract values
ratios = df.iloc[:, 0]
dominant_eigs = df.iloc[:, 1]

# Reverse the data
ratios_reversed = ratios[::-1]
dominant_eigenvalues_reversed = dominant_eigs[::-1]

# Create scatter plot: valid data ratio vs. dominant eigenvalue magnitude
plt.figure(figsize=(8, 6))
plt.scatter(ratios_reversed, dominant_eigenvalues_reversed, color='b', label='Dominant Eigenvalue')
plt.xlabel('Valid Data Ratio')
plt.ylabel('Dominant Eigenvalue Magnitude')
plt.title('Dominant Eigenvalue vs. Valid Data Ratio')
plt.grid(True)
plt.legend()
plt.show()