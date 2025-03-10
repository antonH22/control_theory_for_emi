import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ctrl import discrete_optimal_control as doc
from ctrl import utils
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt


emas = ['EMA_mood', 'EMA_disappointed', 'EMA_scared', 'EMA_worry', 'EMA_down', 'EMA_sad', 'EMA_confidence', 'EMA_stress', 'EMA_lonely', 'EMA_energetic', 'EMA_concentration', 'EMA_resilience', 'EMA_tired', 'EMA_satisfied', 'EMA_relaxed']
emis = ['interactive1', 'interactive2', 'interactive3', 'interactive4', 'interactive5', 'interactive6', 'interactive7', 'interactive8']

data_folder = "data"
subfolders = ["MRT1/processed_csv_no_con","MRT2/processed_csv_no_con","MRT3/processed_csv_no_con"]

# Set parameters
split = 1.0
num_rows_threshold = 50 # One file is excluded
length_reduction = 20

# Write results to this file
filename = "trainlen-eigenvaluesA.csv"

def get_training_set(trainlen, X_train, U_train):
    number_to_remove = len(X_train) - trainlen
    if number_to_remove < 0:
        return False
    X_train_new = X_train[number_to_remove:]
    U_train_new = U_train[number_to_remove:]
    return X_train_new, U_train_new

def compute_eigenvalues(X_train_ratio, U_train_ratio):
    A, B, lmbda = utils.stable_ridge_regression(X_train_ratio, U_train_ratio)
    eigenvalues = np.linalg.eigvals(A)
    return eigenvalues

# Dictionary to store ratios and corresponding frobenius norms
results_trainlens = []
results_eigenvalues = []

dataset_list = []
files = []
for subfolder in subfolders:
    folder_path = os.path.join(data_folder, subfolder, "*.csv")
    for file in glob.glob(folder_path):
        df = pd.read_csv(file)
        data = utils.csv_to_dataset(file, emas, emis, invert_columns=[])
        dataset_list.append(data)
        files.append(file)

skip_files = {}
num_analysed_files = 0
for idx, dataset in enumerate(dataset_list):
    if files[idx] in skip_files:
        continue
    X, U = dataset['X'], dataset['Inp']

    if len(X) < num_rows_threshold:
        continue
    
    # Determine the split index for the training and testing data
    valid = ~np.isnan(X).any(axis=1)
    valid_rows = valid[:-1] & valid[1:] # valid rows are those where the predictor and target are both valid (no NaN values)
    total = valid_rows.sum() # total number of valid rows
    split_index = np.searchsorted(np.cumsum(valid_rows), total * split) # searches for the index in (np.cumsum(pairs)) where the cumulative sum first exceeds 70% of the valid rows

    num_analysed_files += 1
    print(f'Loop {num_analysed_files}/176')
    #print('-' * 40)
    #print(f'{files[idx]}:')
    #print(f'Split index: {split_index}')
    
    # Split data
    X_train, X_test = X[:split_index], X[split_index:]
    U_train, U_test = U[:split_index], U[split_index:]

    initial_trainlen = len(X_train)
    skip = False
    eigenvalues = compute_eigenvalues(X_train, U_train)
    dominant_eigenvalue = np.max(np.abs(eigenvalues))
    results_trainlens.append(initial_trainlen)
    results_eigenvalues.append(dominant_eigenvalue)
    print(f'Initial Trainlen: {initial_trainlen}; real_trainlen: {len(X_train)}; dominant eigenvalue: {dominant_eigenvalue}')
    #print(f'Valid ratio train {valid_train_ratio}')
    iterations = 1
    target_trainlen = initial_trainlen - length_reduction
    while target_trainlen > 10:
        # Do multiple ratioed dataset creations -> multiple different removel processes -> decrease standard deviation caused by individual random removal process
        target_trainlen = initial_trainlen - length_reduction * iterations
        if target_trainlen < 1:
            break
        X_train_trainlen, U_train_trainlen = get_training_set(target_trainlen, X_train, U_train)
        eigenvalues = compute_eigenvalues(X_train_trainlen, U_train_trainlen)
        dominant_eigenvalue = np.max(np.abs(eigenvalues))
        results_trainlens.append(target_trainlen)
        results_eigenvalues.append(dominant_eigenvalue)
        #print(f'Trainlen: {target_trainlen}; real_trainlen: {len(X_train_trainlen)}; dominant eigenvalue: {dominant_eigenvalue}')
        iterations += 1

from collections import Counter
element_counts = Counter(results_trainlens)
for element, count in element_counts.items():
    print(f"{element}: {count} times")
        
# Convert results to a DataFrame to save them to a csv file
df_final = pd.DataFrame({
    "trainlen": results_trainlens,
    "eigenvalues": results_eigenvalues,
})

# Save to CSV
filepath = os.path.join("results_trainlen", filename)
df_final.to_csv(filepath, index=False)
print(f'Final results saved to {filename}')

print(f'Datasets analysed {num_analysed_files}')
print(f'Number of datapoints: {len(results_trainlens)}')

# Reverse the data
trainlens_reversed = results_trainlens[::-1]
dominant_eigenvalues_reversed = results_eigenvalues[::-1]

# Create scatter plot: valid data ratio vs. dominant eigenvalue magnitude
plt.figure(figsize=(8, 6))
plt.scatter(trainlens_reversed, dominant_eigenvalues_reversed, color='b', label='Dominant Eigenvalue')
plt.xlabel('Valid Data Ratio')
plt.ylabel('Dominant Eigenvalue Magnitude')
plt.title('Dominant Eigenvalue vs. Valid Data Ratio')
plt.grid(True)
plt.legend()
plt.show()