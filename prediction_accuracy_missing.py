from ctrl import discrete_optimal_control as doc
from ctrl import utils
from investigate_missing_data import compute_missing_data_percentage
import numpy as np
import os
import glob
import pandas as pd

dataset_list = []

emas = ['EMA_mood', 'EMA_disappointed', 'EMA_scared', 'EMA_worry', 'EMA_down', 'EMA_sad', 'EMA_confidence', 'EMA_stress', 'EMA_lonely', 'EMA_energetic', 'EMA_concentration', 'EMA_resilience', 'EMA_tired', 'EMA_satisfied', 'EMA_relaxed']
emis = ['interactive1', 'interactive2', 'interactive3', 'interactive4', 'interactive5', 'interactive6', 'interactive7', 'interactive8']

prep_data_folder = "prep_data"
subfolders = ["MRT1","MRT2","MRT3"]
files = []

# Set the threshold for missing data and the number of valid rows
missing_data_threshold = 1.0
valid_rows_threshold = 50

for subfolder in subfolders:
    folder_path = os.path.join(prep_data_folder, subfolder, "*.csv")
    for file in glob.glob(folder_path):
        df = pd.read_csv(file)
        missing_data_percentage = compute_missing_data_percentage(df)
        if missing_data_percentage < missing_data_threshold:
            data = utils.csv_to_dataset(file, emas, emis, [])
            dataset_list.append(data)
            files.append(file)

mse_list = []
baseline_mse_list = []
mae_per_ema_list = []
mae_list = []

mac_per_ema_list = []
mac_list = []
msc_list = []

high_mse_files = []
skip_files = {}
num_analysed_files = 0
for idx, dataset in enumerate(dataset_list):
    if files[idx] in skip_files:
        continue

    X, U = dataset['X'], dataset['Inp']
    
    # Determine the split index for the training and testing data
    valid = ~np.isnan(X).any(axis=1)
    valid_rows = valid[:-1] & valid[1:] # valid rows are those where the predictor and target are both valid (no NaN values)
    total = valid_rows.sum() # total number of valid rows
    split_index = np.searchsorted(np.cumsum(valid_rows), total * 0.7) # searches for the index in (np.cumsum(pairs)) where the cumulative sum first exceeds or equals 70
    # Leads to lower standard deviation of the mean squared error and mean absolute error (compared to split_index = int(0.7 * len(X)))

    # Skip if there are less than  valid row
    if total < valid_rows_threshold:
        continue

    num_analysed_files += 1
    
    # Split data
    X_train, X_test = X[:split_index], X[split_index:]
    U_train, U_test = U[:split_index], U[split_index:]

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
    mse_list.append(mse)

    # Compute the baseline mean squared error of the test data
    target_mean_per_ema = np.mean(real_target_states, axis = 0)
    target_mean_array = np.full_like(real_target_states, fill_value=target_mean_per_ema)
    baseline_mse = np.mean((target_mean_array - real_target_states)**2)
    baseline_mse_list.append(baseline_mse)

    # Compute the Mean Absolute Error (MAE)
    absolute_differences = np.abs(predicted_states - real_target_states)
    mean_absolute_errors = np.mean(absolute_differences, axis=0)
    mae_per_ema_list.append(mean_absolute_errors)

    mae = np.mean(mean_absolute_errors)
    mae_list.append(mae)

    # Compute the mean absolute change (MAC) of the state per time step
    real_absolute_differences = np.abs(real_predictor_states - real_target_states)
    mean_real_abs_differences = np.mean(real_absolute_differences, axis = 0)
    mac_per_ema_list.append(mean_real_abs_differences)

    mac = np.mean(mean_real_abs_differences)
    mac_list.append(mac)

    # Compute the mean squared change (MSC) of the state per time step
    real_squared_differences = np.square(real_absolute_differences)
    msc = np.mean(real_squared_differences)
    msc_list.append(msc)

    num_interventions_train = U_train.sum()
    num_interventions_test = U_test.sum()

    # Append to files_mean_error_greater_10 if mse > 10 to filter out bad datasets
    if mse > 4:
        high_mse_files.append(files[idx])
    
    # Print results
    print('-' * 40)
    print(f'{files[idx]}:')
    print(f'Number of valid rows: {total}')
    print(f'Split index (70%): {split_index}')
    print(f'MSE: {np.round(mse,2)}')
    print(f'MSC: {np.round(msc,2)}')
    print(f'Baseline MSE: {np.round(baseline_mse,2)}')
    print(f'MAE: {np.round(mae,2)}')
    print(f'MAC: {np.round(mac,2)}')
    #Print the MAE for each variable
    #df_mae_per_variable = pd.DataFrame({'MAE': np.round(mean_absolute_errors,2), 'real': np.round(mean_real_abs_differences,2)}, index=emas)
    #print(df_mae_per_variable)
    #print(f'Number of interventions in training data: {num_interventions_train}')
    #print(f'Number of interventions in testing data: {num_interventions_test}')
    
mse_array = np.array(mse_list)
max_mse = np.max(mse_array)
min_mse = np.min(mse_array)
mean_mse = np.mean(mse_array)
std_mse = np.std(mse_array)

baseline_mse_array = np.array(baseline_mse_list)
mean_baseline_mse = np.mean(baseline_mse_array)

mae_array = np.array(mae_list)
max_mae = np.max(mae_array)
min_mae = np.min(mae_array)
mean_mae = np.mean(mae_array)
std_mae = np.std(mae_array)

mae_per_ema_array= np.array(mae_per_ema_list)
mean_mae_per_ema = np.mean(mae_per_ema_array, axis=0)

mac_per_ema_array = np.array(mac_per_ema_list)
mean_mac_per_ema = np.mean(mac_per_ema_array, axis=0)
mean_mac = np.mean(mean_mac_per_ema)

msc_array = np.array(msc_list)
mean_msc = np.mean(msc_array)

df_mean_mae = pd.DataFrame({
    'MAE': np.round(mean_mae_per_ema,2),
    'MAC': np.round(mean_mac_per_ema,2)
}, index = emas)

print()
print('#' * 40)
print()
print(f'Number of datasets analysed: {num_analysed_files}')
#print(f'Datasets with MSE > 4: {high_mse_files}') # to filter out bad datasets
print()
#print(f'Max MSE: {max_mse}')
#print(f'Min MSE: {min_mse}')
print(f'Mean MSE: {np.round(mean_mse,2)} ({np.round(std_mse,2)})')
print(f'Mean MSC: {np.round(mean_msc,2)}')
print(f'Mean Baseline MSE: {np.round(mean_baseline_mse,2)}')
print()
#print(f'Max MAE: {max_mae}')
#print(f'Min MAE: {min_mae}')
print(f'Mean MAE: {np.round(mean_mae,2)} ({np.round(std_mae,2)})')
print(f'Mean MAC: {np.round(mean_mac,2)}')
print()
print(f'MAE and MAC per EMA:')
print(df_mean_mae)