from ctrl import discrete_optimal_control as doc
from ctrl import utils
from investigate_missing_data import compute_missing_data_percentage
import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

dataset_list = []

emas = ['EMA_mood', 'EMA_disappointed', 'EMA_scared', 'EMA_worry', 'EMA_down', 'EMA_sad', 'EMA_confidence', 'EMA_stress', 'EMA_lonely', 'EMA_energetic', 'EMA_concentration', 'EMA_resilience', 'EMA_tired', 'EMA_satisfied', 'EMA_relaxed']
emis = ['interactive1', 'interactive2', 'interactive3', 'interactive4', 'interactive5', 'interactive6', 'interactive7', 'interactive8']

data_folder = "data"
subfolders = ["MRT1/processed_csv_no_con","MRT2/processed_csv_no_con","MRT3/processed_csv_no_con"]
files = []

# Set the threshold for missing data and the number of valid rows
num_valid_training_rows = 50
valid_rows_threshold = 70
max_len = 200

for subfolder in subfolders:
    folder_path = os.path.join(data_folder, subfolder, "*.csv")
    for file in glob.glob(folder_path):
        df = pd.read_csv(file)
        data = utils.csv_to_dataset(file, emas, emis, invert_columns=[])
        dataset_list.append(data)
        files.append(file)

def compute_max_len(files):
    max_len = 0
    max_file = ''
    for file in files:
        df = pd.read_csv(file)
        num_valid_rows = ((df.notna() & df.shift(-1).notna()).all(axis=1)).sum() - 1
        if num_valid_rows > max_len:
            max_len = num_valid_rows
            max_file = file
    print(f'Maximum number of valid rows: {max_len} in {max_file}')
    return max_len

mean_per_ema_list = []

mse_list = []
baseline_mse_list = []
mae_per_ema_list = []
mae_list = []

mac_per_ema_list = []
mac_list = []
msc_list = []

mse_per_step_overall_list = []
mae_per_step_overall_list = []

high_mse_files = []
skip_files = {}
num_analysed_files = 0
for idx, dataset in enumerate(dataset_list):
    if files[idx] in skip_files:
        continue

    X, U = dataset['X'], dataset['Inp']

    mean_per_ema = np.nanmean(X, axis=0)
    mean_per_ema_list.append(mean_per_ema)
    
    # Determine the split index for the training and testing data
    valid = ~np.isnan(X).any(axis=1)
    valid_rows = valid[:-1] & valid[1:] # valid rows are those where the predictor and target are both valid (no NaN values)
    total = valid_rows.sum() # total number of valid rows
    split_index = np.searchsorted(np.cumsum(valid_rows), num_valid_training_rows) # searches for the index in (np.cumsum(pairs)) where the cumulative sum first exceeds or equals 50

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

    mse_per_step_list = []
    mae_per_step_list = []

    # Prediction loop for the training data 
    for i in range(len(X_train) -1):
        # Skip if there is a NaN value in the training data in predictor or target
        if np.isnan(X_train[i]).any() or np.isnan(X_train[i + 1]).any():
            continue
        real_predictor_states.append(X_train[i])
        real_target_states.append(X_train[i+1])
        x_next = doc.step(A, B, X_train[i], U_train[i])
        predicted_states.append(x_next)

        # Compute the MSE and MAE per time step
        mse_per_step = np.mean((x_next - X_train[i+1])**2)
        mae_per_step = np.mean(np.abs(x_next - X_train[i+1]))
        mse_per_step_list.append(mse_per_step)
        mae_per_step_list.append(mae_per_step)
    
    # Prediction loop for the testing data
    for i in range(len(X_test) -1):
        # Include NaN data in the training set so that ridge regression to disregard non valid rows
        X_train = np.append(X_train, X_test[i-1:i], axis=0)
        U_train = np.append(U_train, U_test[i-1:i], axis=0)
        # Skip if there is a NaN value in the test data in predictor or target
        if np.isnan(X_test[i]).any() or np.isnan(X_test[i + 1]).any():
            continue
        real_predictor_states.append(X_test[i])
        real_target_states.append(X_test[i+1])
        A , B, lmbda = utils.stable_ridge_regression(X_train, U_train)
        x_next = doc.step(A, B, X_test[i], U_test[i])
        predicted_states.append(x_next)

        # Compute the MSE and MAE per time step
        mse_per_step = np.mean((x_next - X_test[i+1])**2)
        mae_per_step = np.mean(np.abs(x_next - X_test[i+1]))
        mse_per_step_list.append(mse_per_step)
        mae_per_step_list.append(mae_per_step)

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

    while len(mse_per_step_list) < max_len:
        mse_per_step_list.append(np.nan)

    while len(mae_per_step_list) < max_len:
        mae_per_step_list.append(np.nan)    

    mse_per_step_overall_list.append(mse_per_step_list[:max_len])
    mae_per_step_overall_list.append(mae_per_step_list[:max_len])

    # Append to files_mean_error_greater_10 if mse > 10 to filter out bad datasets
    if mse > 4:
        high_mse_files.append(files[idx])
    
    # Print results
    print('-' * 40)
    print(f'{files[idx]}:')
    print(f'Number of valid rows: {total}')
    print(f'Split index: {split_index}')
    """
    print(f'MSE: {np.round(mse,2)}')
    print(f'MSC: {np.round(msc,2)}')
    print(f'Baseline MSE: {np.round(baseline_mse,2)}')
    print(f'MAE: {np.round(mae,2)}')
    print(f'MAC: {np.round(mac,2)}')
    """
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

mean_per_ema_array = np.array(mean_per_ema_list)
mean_per_ema = np.mean(mean_per_ema_array, axis=0)
mean_ema_overall = np.mean(mean_per_ema)
mean_ema_overall2 = np.mean(mean_per_ema_array)

msc_array = np.array(msc_list)
mean_msc = np.mean(msc_array)

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
df_mean_mae = pd.DataFrame({
    'MAE': np.round(mean_mae_per_ema,2),
    'MAC': np.round(mean_mac_per_ema,2)
}, index = emas)
print(df_mean_mae)
print()
df_mean_ema = pd.DataFrame({
    'mean': np.round(mean_per_ema,2)
}, index = emas)
#print(df_mean_ema)
#print(f'Mean overall EMA: {np.round(mean_ema_overall,2)}')

mse_per_step_overall_array = np.array(mse_per_step_overall_list)
mae_per_step_overall_array = np.array(mae_per_step_overall_list)
mse_per_step_overall_mean = np.nanmean(mse_per_step_overall_array, axis=0)
mae_per_step_overall_mean = np.nanmean(mae_per_step_overall_array, axis=0)

# Create a plot
plt.plot(mae_per_step_overall_mean, linestyle='-', color='blue', label='MAE')

# Add titles and labels
plt.title("MAE per Step (Across datasets with >= 70 valid rows)")
plt.xlabel("Step")
plt.ylabel("MAE")

# Display the plot
plt.show()