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

# Collect all CSV files with less than 50% missing rows from the specified subfolders
prep_data_folder = "prep_data"
subfolders = ["MRT1","MRT2","MRT3"]
files = []
missing_data_threshold = 0.7
num_analysed_files = 0

for subfolder in subfolders:
    folder_path = os.path.join(prep_data_folder, subfolder, "*.csv")
    for file in glob.glob(folder_path):
        df = pd.read_csv(file)
        missing_data_percentage = compute_missing_data_percentage(df)
        if missing_data_percentage < missing_data_threshold:
            data = utils.csv_to_dataset(file, emas, emis, [])
            dataset_list.append(data)
            files.append(file)

mean_squared_error_list = []
mean_absolute_error_per_var_list = []
mean_absolute_error_list = []
mean_real_absolute_difference_per_var_list = []
mean_real_mae_list = []
mean_real_mse_list = []
high_mean_error_files = []

skip_files = {}
for idx, dataset in enumerate(dataset_list):
    if files[idx] in skip_files:
        continue
    num_analysed_files += 1
    X, U = dataset['X'], dataset['Inp']
    
    # Determine the split index for the training and testing data
    valid = ~np.isnan(X).any(axis=1)
    valid_rows = valid[:-1] & valid[1:] # valid rows are those where the predictor and target are both valid (no NaN values)
    total = valid_rows.sum() # total number of useful rows
    split_index = np.searchsorted(np.cumsum(valid_rows), total * 0.7) # searches for the index in (np.cumsum(pairs)) where the cumulative sum first exceeds or equals 70%
    # Leads to lower standard deviation of the mean squared error and mean absolute error (compared to split_index = int(0.7 * len(X)))
    
    # Split data
    X_train, X_test = X[:split_index], X[split_index:]
    U_train, U_test = U[:split_index], U[split_index:]

    # Handle missing data using Last Observation Carried Forward (LOCF) method
    df_helper_locf = pd.DataFrame(X_train)
    df_helper_locf.fillna(method='ffill', inplace=True)
    X_train = df_helper_locf.to_numpy()

    # Infer the A and B matrices using stable ridge regression
    A, B, lmbda = utils.stable_ridge_regression(X_train, U_train) # the lmbda output is the regularization parameter that is used to render A stable
    
    # Predict the next state using the inferred A and B matrices
    predicted_states = []
    real_predictor_states = [] # To keep track of rows that are skipped in the prediction loop
    real_target_states = []

    #print('Timestep | Predictor | Target | Prediction  (mood)')
    for i in range(len(X_test) -1):
        # Skip if there is a NaN value in the test data in predictor or target
        if np.isnan(X_test[i]).any() or np.isnan(X_test[i + 1]).any():
            continue
        real_predictor_states.append(X_test[i])
        real_target_states.append(X_test[i+1])
        x_next = doc.step(A, B, X_test[i], U_test[i])
        predicted_states.append(x_next)
        
        # To see that it works correctly
        timestep = i + 2 + split_index
        #print(timestep, X_test[i][0], X_test[i+1][0], np.round(x_next[0],2))

    predicted_states = np.array(predicted_states)
    real_predictor_states = np.array(real_predictor_states)
    real_target_states = np.array(real_target_states)

    # Compute the mean squared error
    mse = np.mean((predicted_states - real_target_states)**2)
    mean_squared_error_list.append(mse)

    # Compute the Mean Absolute Error (MAE)
    absolute_differences = np.abs(predicted_states - real_target_states)
    mean_absolute_errors = np.mean(absolute_differences, axis=0)
    mean_absolute_error_per_var_list.append(mean_absolute_errors)

    mae = np.mean(mean_absolute_errors)
    mean_absolute_error_list.append(mae)

    # Compute the (real) mean absolute differences of the state per time step
    real_absolute_differences = np.abs(real_predictor_states - real_target_states)
    mean_real_abs_differences = np.mean(real_absolute_differences, axis = 0)
    mean_real_absolute_difference_per_var_list.append(mean_real_abs_differences)

    real_mae = np.mean(mean_real_abs_differences)
    mean_real_mae_list.append(real_mae)

    # Compute the (real) mean squared differences of the state per time step
    squared_differences = np.square(real_absolute_differences)
    mse_per_variable = np.mean(squared_differences, axis=0)
    real_mse = np.mean(squared_differences)
    mean_real_mse_list.append(real_mse)

    num_interventions_train = U_train.sum()
    num_interventions_test = U_test.sum()

    # Append to files_mean_error_greater_10 if mse > 10 to filter out bad datasets
    if mse > 10:
        high_mean_error_files.append(files[idx])
    
    # Print results
    print('-' * 40)
    print(f'{files[idx]}:')
    print(f'Number of useful rows: {total}')
    print(f'Split index (70%): {split_index}')
    print(f'Mean Squared Error: {np.round(mse,2)} ({np.round(real_mse,2)})')
    print(f'Mean Absolute Error: {np.round(mae,2)} ({np.round(real_mae,2)})')
    # Print the MAE for each variable
    df_mae_per_variable = pd.DataFrame({'MAE': np.round(mean_absolute_errors,2), 'real': np.round(mean_real_abs_differences,2)}, index=emas)
    print(df_mae_per_variable)
    #print(f'Number of interventions in training data: {num_interventions_train}')
    #print(f'Number of interventions in testing data: {num_interventions_test}')
    
mean_squared_errors = np.array(mean_squared_error_list)
max_mean_squared_error = np.max(mean_squared_errors)
min_mean_squared_error = np.min(mean_squared_errors)
mean_mean_squared_error = np.mean(mean_squared_errors)
std_mean_squared_error = np.std(mean_squared_errors)

mean_absolute_errors = np.array(mean_absolute_error_list)
max_mean_absolute_error = np.max(mean_absolute_errors)
min_mean_absolute_error = np.min(mean_absolute_errors)
mean_mean_absolute_error = np.mean(mean_absolute_errors)
std_mean_absolute_error = np.std(mean_absolute_errors)

mean_absolute_errors_per_var = np.array(mean_absolute_error_per_var_list)
mean_mean_abs_error_per_var = np.mean(mean_absolute_errors_per_var, axis=0)

mean_differences_per_var = np.array(mean_real_absolute_difference_per_var_list)
mean_mean_differences_per_var = np.mean(mean_differences_per_var, axis=0)

mean_real_mses = np.array(mean_real_mse_list)
mean_mean_mses = np.mean(mean_real_mses)

df_mean_mae = pd.DataFrame({
    'MAE': np.round(mean_mean_abs_error_per_var,2),
    'real': np.round(mean_mean_differences_per_var,2)
}, index = emas)

print()
print('#' * 40)
print()
print(f'Number of datasets analysed: {num_analysed_files}')
print(f'Datasets with Mean Squared Error > 10: {high_mean_error_files}') # to filter out bad datasets
print()
#print(f'Max Mean Squared Error: {max_mean_squared_error}')
#print(f'Min Mean Squared Error: {min_mean_squared_error}')
print(f'Mean Mean Squared Error: {np.round(mean_mean_squared_error,2)} ({np.round(mean_mean_mses,2)})')
print(f'Std Mean Squared Error: {np.round(std_mean_squared_error,2)}')
print()
#print(f'Max Mean Absolute Error: {max_mean_absolute_error}')
#print(f'Min Mean Absolute Error: {min_mean_absolute_error}')
print(f'Mean Mean Absolute Error: {np.round(mean_mean_absolute_error,2)} ({np.round(np.mean(mean_mean_differences_per_var),2)})')
print(f'Std Mean Absolute Error: {np.round(std_mean_absolute_error,2)}')
print()
print(f'Mean Absolute Error (model prediction) vs real mean differences per variable:')
print(df_mean_mae)