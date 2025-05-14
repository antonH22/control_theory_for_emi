import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ctrl import discrete_optimal_control as doc
from ctrl import utils
from enhanced_eval_ctrl_approach import myutils
import numpy as np
import pandas as pd
import json

### Evaluating missing data handling strategies (5.1.3, Table 4)

emas = ['EMA_mood', 'EMA_disappointed', 'EMA_scared', 'EMA_worry', 'EMA_down', 'EMA_sad', 'EMA_confidence', 'EMA_stress', 'EMA_lonely', 'EMA_energetic', 'EMA_concentration', 'EMA_resilience', 'EMA_tired', 'EMA_satisfied', 'EMA_relaxed']
emis = ['interactive1', 'interactive2', 'interactive3', 'interactive4', 'interactive5', 'interactive6', 'interactive7', 'interactive8']

data_folder = "data"
subfolders = ["MRT1/processed_csv_no_con", "MRT2/processed_csv_no_con", "MRT3/processed_csv_no_con"]

dataset_list, files = myutils.load_dataset(data_folder, subfolders, emas, emis, centered=True)

splits = [0.7, 0.75, 0.8, 0.85]

def locf(X_train):
    " Imputes missing values using Last Observation Carried Forward (LOCF). "
    df_helper_locf = pd.DataFrame(X_train).copy()
    df_helper_locf.ffill(inplace=True)
    X_train_locf = df_helper_locf.to_numpy()
    return X_train_locf

def mean_imputation(X_train):
    " Imputes missing values with column means. "
    df_helper_mean = pd.DataFrame(X_train)
    df_helper_mean.fillna(df_helper_mean.mean(), inplace=True)
    X_train_mean = df_helper_mean.to_numpy()
    return X_train_mean

# Set the threshold for the number of valid rows (50-th highest = 91, 30-th highest=128; determined via dataset_statistics.py script)
valid_rows_threshold = 0

# Select the imputation function (None for no imputation)
imputation_fun = locf

mse_list = []
baseline_mse_list = []
mae_per_ema_list = []
mae_list = []
mac_per_ema_list = []
mac_list = []
msc_list = []

dataset_list, files = myutils.load_dataset(data_folder, subfolders, emas, emis, centered=True)

with open('exclude_participant_list.json', 'r') as f:
    exclude_participant_list = json.load(f)

num_analysed_files = 0
for dataset, file in zip(dataset_list, files):
    if file in exclude_participant_list:
        # Participants for whom there is no trained RNN model are excluded from all analyses
        continue

    X, U = dataset['X'], dataset['Inp']

    valid_rows = myutils.get_valid_rows(X)
    num_valid_rows = valid_rows.sum()

    # Skip if there are less than x valid row (e.g., to evaluate top 50 participants)
    if num_valid_rows <= valid_rows_threshold:
        continue
    
    num_analysed_files += 1
    threshold_map = {0: 143, 91: 50, 128: 30}
    print(f'Iteration {num_analysed_files}/{threshold_map.get(valid_rows_threshold, "?")}')

    mse_split_list = []
    mae_split_list = []
    mac_split_list = []
    msc_split_list = []
    baseline_mse_split_list = []

    for split in splits:
        # Determine the split index for the training and testing data
        split_index = np.searchsorted(np.cumsum(valid_rows), num_valid_rows * split) # searches for the index in (np.cumsum(pairs)) where the cumulative sum first exceeds 70% of the valid rows
        # Leads to lower standard deviation of the mean squared error and mean absolute error (compared to split_index = int(0.7 * len(X)))

        X_train, X_test = X[:split_index], X[split_index:]
        U_train, U_test = U[:split_index], U[split_index:]

        if imputation_fun is not None:
            X_train = imputation_fun(X_train)

        # Infer the A and B matrices using stable ridge regression
        A, B, lmbda = utils.stable_ridge_regression(X_train, U_train)
        
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

            A, B, lmbda = utils.stable_ridge_regression(X_train, U_train)
            x_next = doc.step(A, B, X_test[i], U_test[i])
            predicted_states.append(x_next)

        predicted_states = np.array(predicted_states)
        real_target_states = np.array(real_target_states)

        # Compute the mean squared error
        mse_split_list.append(np.mean((predicted_states - real_target_states)**2))

        # Compute the baseline mean squared error of the test data
        target_mean_per_ema = np.mean(real_target_states, axis = 0)
        target_mean_array = np.full_like(real_target_states, fill_value=target_mean_per_ema)
        baseline_mse_split_list.append(np.mean((target_mean_array - real_target_states)**2))

        # Compute the Mean Absolute Error (MAE)
        absolute_differences = np.abs(predicted_states - real_target_states)
        mean_absolute_errors = np.mean(absolute_differences, axis=0)
        mae_split_list.append(np.mean(mean_absolute_errors))

        # Compute the mean absolute change (MAC) of the state per time step
        real_absolute_differences = np.abs(real_predictor_states - real_target_states)
        mean_real_abs_differences = np.mean(real_absolute_differences, axis = 0)
        mac_split_list.append(np.mean(mean_real_abs_differences))

        # Compute the mean squared change (MSC) of the state per time step
        real_squared_differences = np.square(real_absolute_differences)
        msc_split_list.append(np.mean(real_squared_differences))

    # Average over splits
    mse = np.mean(mse_split_list)
    baseline_mse = np.mean(baseline_mse_split_list)
    mae = np.mean(mae_split_list)
    mac = np.mean(mac_split_list)
    msc = np.mean(msc_split_list)

    mse_list.append(mse)
    baseline_mse_list.append(baseline_mse)
    mae_list.append(mae)
    mac_list.append(mac)
    msc_list.append(msc)
    mae_per_ema_list.append(mean_absolute_errors)
    mac_per_ema_list.append(mean_real_abs_differences)

    num_interventions_train = U_train.sum()
    num_interventions_test = U_test.sum()

    # Print individual results
    """
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
    """
    
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
print()
#print(f'Max MSE: {max_mse}')
#print(f'Min MSE: {min_mse}')
print(f'Mean MSE: {np.round(mean_mse,2)} ({np.round(std_mse,2)})')
print(f'Mean MSC: {np.round(mean_msc,2)}')
print(f'Mean Baseline MSE: {np.round(mean_baseline_mse,2)}')
print()
print(f'Max MAE: {max_mae}')
print(f'Min MAE: {min_mae}')
print(f'Mean MAE: {np.round(mean_mae,2)} ({np.round(std_mae,2)})')
print(f'Mean MAC: {np.round(mean_mac,2)}')
print()
print(f'MAE and MAC per EMA:')
print(df_mean_mae)