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
# Loop through all datasets in dataset_list
skip_files = {'prep_data\\MRT1\\11228_120_prep.csv', 'prep_data\\MRT2\\12600_19_prep.csv', 'prep_data\\MRT2\\12600_41_prep.csv', 'prep_data\\MRT2\\12600_63_prep.csv', 'prep_data\\MRT3\\12600_227_prep.csv', 'prep_data\\MRT3\\12600_238_prep.csv', 'prep_data\\MRT3\\12600_270_prep.csv'}
for idx, dataset in enumerate(dataset_list):
    if files[idx] in skip_files:
        continue

    X, U = dataset['X'], dataset['Inp']
    
    valid = ~np.isnan(X).any(axis=1)
    pairs = valid[:-1] & valid[1:]
    total = pairs.sum()
    split_index = np.searchsorted(np.cumsum(pairs), total * 0.7)
    
    # Split EMA data
    X_train, X_test = X[:split_index], X[split_index:]
    
    # Split input data
    U_train, U_test = U[:split_index], U[split_index:]
    
    # Store the split data back into the dataset
    dataset['X_train'], dataset['X_test'] = X_train, X_test
    dataset['U_train'], dataset['U_test'] = U_train, U_test
    
    
    # Infer the A and B matrices using stable ridge regression
    A, B, lmbda = utils.stable_ridge_regression(X_train, U_train)     # the lmbda output is the regularization parameter that is used to render A stable
    
    mask_nan = np.isnan(X_test).any(axis=1)
    X_test = X_test[~mask_nan]
    U_test = U_test[~mask_nan]
    
    states = []
    for i in range(len(X_test)):
        x_next = doc.step(A, B, X_test[i], U_test[i])
        states.append(x_next)
    
    states = np.array(states)
    states = states[:-1]
    X_test = X_test[1:]

    mse = np.mean((states - X_test)**2)
    mean_squared_errors.append(mse)
    if mse > 10:
        files_mean_error_greater_10.append(files[idx])
    # Print results
    print(f'{files[idx]}:')
    print(f'Number of valid predictor pairs: {total}')
    print(f'Split index (70%): {split_index}')
    print(f'Mean Squared Error: {mse}')
    print('-' * 40)

mean_squared_errors = np.array(mean_squared_errors)
max_mean_squared_error = np.max(mean_squared_errors)
min_mean_squared_error = np.min(mean_squared_errors)
mean_mean_squared_error = np.mean(mean_squared_errors)
std_mean_squared_error = np.std(mean_squared_errors)

print(f'Number of datasets analysed: {len(dataset_list)}')
#print(f'Datasets with Mean Squared Error > 10: {files_mean_error_greater_10}') to filter out bad datasets
print(f'Max Mean Squared Error: {max_mean_squared_error}')
print(f'Min Mean Squared Error: {min_mean_squared_error}')
print(f'Mean Mean Squared Error: {mean_mean_squared_error}')
print(f'Std Mean Squared Error: {std_mean_squared_error}')
