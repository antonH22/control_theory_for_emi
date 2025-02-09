from ctrl import discrete_optimal_control as doc
from ctrl import utils
from investigate_missing_data import compute_missing_data_percentage
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd

dataset_list = []

emas = ['EMA_mood', 'EMA_disappointed', 'EMA_scared', 'EMA_worry', 'EMA_down', 'EMA_sad', 'EMA_confidence', 'EMA_stress', 'EMA_lonely', 'EMA_energetic', 'EMA_concentration', 'EMA_resilience', 'EMA_tired', 'EMA_satisfied', 'EMA_relaxed']
emis = ['interactive1', 'interactive2', 'interactive3', 'interactive4', 'interactive5', 'interactive6', 'interactive7', 'interactive8']
invert_columns = ['EMA_mood', 'EMA_confidence', 'EMA_energetic', 'EMA_concentration', 'EMA_resilience', 'EMA_satisfied', 'EMA_relaxed']

# Collect all CSV files with less than 50% missing rows from the specified subfolders
prep_data_folder = "prep_data"
subfolders = ["MRT1","MRT2","MRT3"]
files = []
missing_data_threshold = 0.7

for subfolder in subfolders:
    folder_path = os.path.join(prep_data_folder, subfolder, "*.csv")
    for file in glob.glob(folder_path):
        df = pd.read_csv(file)
        missing_data_percentage = compute_missing_data_percentage(df)
        if missing_data_percentage < missing_data_threshold:
            data = utils.csv_to_dataset(file, emas, emis, invert_columns)
            dataset_list.append(data)
            files.append(file)

mean_squared_errors = []
files_mean_error_greater_10 = []
skip_files = {'prep_data\\MRT2\\12600_16_prep.csv', 'prep_data\\MRT2\\12600_19_prep.csv', 'prep_data\\MRT2\\12600_63_prep.csv', 'prep_data\\MRT3\\12600_227_prep.csv', 'prep_data\\MRT3\\12600_238_prep.csv', 'prep_data\\MRT3\\12600_270_prep.csv'}
for idx, dataset in enumerate(dataset_list):
    if files[idx] in skip_files:
        continue
    X, U = dataset['X'], dataset['Inp']
    
    # Determine the split index for the training and testing data
    valid = ~np.isnan(X).any(axis=1)
    pairs = valid[:-1] & valid[1:]
    total = pairs.sum()
    split_index = np.searchsorted(np.cumsum(pairs), total * 0.7)
    
    # Split data
    X_train, X_test = X[:split_index], X[split_index:]
    U_train, U_test = U[:split_index], U[split_index:]

    # Infer the A and B matrices using stable ridge regression
    A, B, lmbda = utils.stable_ridge_regression(X_train, U_train)     # the lmbda output is the regularization parameter that is used to render A stable
    
    # Predict the next state using the inferred A and B matrices
    states = []
    rows_to_delete = [] # To keep track of rows that are skipped in the prediction loop
    for i in range(len(X_test) -1):
        # Skip if there is a NaN value in the test data (predictor or target)
        if np.isnan(X_test[i]).any() or np.isnan(X_test[i + 1]).any():
            rows_to_delete.append(i)  # Collect the index to delete
            continue
        x_next = doc.step(A, B, X_test[i], U_test[i])
        states.append(x_next)
    
    X_test = np.delete(X_test, rows_to_delete, axis=0)
    X_test = X_test[1:]
    states = np.array(states)
    # If the last row contains NaN, remove it from both X_test and states
    if np.isnan(X_test[-1]).any():
        X_test = X_test[:-1]
        states = states[:-1]

    # Compute the mean squared error
    mse = np.mean((states - X_test)**2)
    mean_squared_errors.append(mse)
    # Append to files_mean_error_greater_10 if mse > 10 to filter out bad datasets
    if mse > 10:
        files_mean_error_greater_10.append(files[idx])
    # Print results
    print(f'{files[idx]}:')
    print(f'Number of useful rows: {total}')
    #print(f'Split index (70%): {split_index}')
    print(f'Mean Squared Error: {mse}')
    print('-' * 40)

mean_squared_errors = np.array(mean_squared_errors)
max_mean_squared_error = np.max(mean_squared_errors)
min_mean_squared_error = np.min(mean_squared_errors)
mean_mean_squared_error = np.mean(mean_squared_errors)
std_mean_squared_error = np.std(mean_squared_errors)

print(f'Number of datasets analysed: {len(dataset_list)}')
print(f'Datasets with Mean Squared Error > 10: {files_mean_error_greater_10}') # to filter out bad datasets
print(f'Max Mean Squared Error: {max_mean_squared_error}')
print(f'Min Mean Squared Error: {min_mean_squared_error}')
print(f'Mean Mean Squared Error: {mean_mean_squared_error}')
print(f'Std Mean Squared Error: {std_mean_squared_error}')
