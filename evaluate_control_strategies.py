from ctrl import utils
from ctrl import discrete_optimal_control as doc
from ctrl import control_strategies as strats
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt

import os


emas = ['EMA_mood', 'EMA_disappointed', 'EMA_scared', 'EMA_worry', 'EMA_down', 'EMA_sad', 'EMA_confidence', 'EMA_stress', 'EMA_lonely', 'EMA_energetic', 'EMA_concentration', 'EMA_resilience', 'EMA_tired', 'EMA_satisfied', 'EMA_relaxed']
emis = ['interactive1', 'interactive2', 'interactive3', 'interactive4', 'interactive5', 'interactive6', 'interactive7', 'interactive8']

data_folder = "data"
#subfolders = ["MRT2/processed_csv_no_con"]
subfolders = ["MRT1/processed_csv_no_con","MRT2/processed_csv_no_con","MRT3/processed_csv_no_con"]

num_rows_threshold = 50 # One file is excluded

n_steps = 1
brute_force_time_horizon = 5
rho = 1
skip_files = ["data\MRT1/processed_csv_no_con\\11228_17.csv", "data\\MRT1/processed_csv_no_con\\11228_19.csv", "data\MRT1/processed_csv_no_con\\11228_35.csv","data\MRT2/processed_csv_no_con\\12600_40.csv","data\MRT2/processed_csv_no_con\\12600_42.csv", "data\MRT3/processed_csv_no_con\\12600_201.csv", "data\MRT3/processed_csv_no_con\\12600_203.csv", "data\MRT3/processed_csv_no_con\\12600_236.csv", "data\MRT3/processed_csv_no_con\\12600_239.csv"]
skip_files = []
dataset_list, files = utils.load_dataset(data_folder, subfolders, emas, emis, centered=True, exclude_constant_columns=True)

def locf(X_train): 
    df_helper_locf = pd.DataFrame(X_train).copy()
    df_helper_locf.ffill(inplace=True)
    X_train_locf = df_helper_locf.to_numpy()
    return X_train_locf

def compute_real_wellbeing_change(data, index, n_steps):
    data = locf(data)
    if index + n_steps >= data.shape[0]:
        return False 
    future_mean = np.mean(data[index + 1 : index + n_steps + 1], axis=0)
    wellbeing_change = future_mean - data[index]
    return wellbeing_change

def predict_n_steps_lds(locf_X, U, index, control_seq_opt_ctrl, n_steps):
    A, B, lmbda = utils.stable_ridge_regression(locf_X, U)
    predictions = []

    U_strategy = U.copy()
    U_strategy[index] = control_seq_opt_ctrl
    
    for i in range(n_steps):
        x_next = doc.step(A, B, locf_X[index + i], U_strategy[index + i])
        predictions.append(x_next)

    return predictions

def mean_and_standard_error(data):
    data = np.array(data)
    mean_value = np.mean(data)
    std_dev = np.std(data)
    n = len(data)
    standard_error = std_dev / np.sqrt(n)
    return mean_value, standard_error

overall_real_wellbeing_change = []
overall_wellbeing_change_opt_ctrl = []
overall_wellbeing_change_brute_force = []
overall_wellbeing_change_max_ac = []

count_nan = 0
for idx, dataset in enumerate(dataset_list):
    if files[idx] in skip_files:
        continue
    X, U = dataset['X'], dataset['Inp']
    if len(X) < num_rows_threshold:
        continue
    
    n_items = X.shape[1]
    n_inputs = U.shape[1]
    target_state = np.full(n_items, -3)
    admissible_inputs = np.eye(n_inputs)

    input_rows_indeces = np.where(~np.all(U == 0, axis=1))[0].tolist()
    print(files[idx])

    real_wellbeing_changes = []
    wellbeing_changes_opt_ctrl = []
    wellbeing_changes_brute_force = []
    wellbeing_changes_max_ac = []
    for index in input_rows_indeces:
        if index < 20:
            continue
        
        real_wellbeing_change = compute_real_wellbeing_change(X, index, n_steps)
        if real_wellbeing_change is False:
            continue  # Skip this index if compute real wellbeing change failed
        real_wellbeing_changes.append(real_wellbeing_change)
        
        locf_X = locf(X)

        input_opt_ctrl = strats.optimal_control_strategy(locf_X, U, target_state, admissible_inputs, rho, online=True)
        predictions_opt_ctrl = predict_n_steps_lds(locf_X, U, index, input_opt_ctrl, n_steps)
        predicted_wellbeing_change_opt_ctrl = np.mean(predictions_opt_ctrl, axis=0) - locf_X[index]
        wellbeing_changes_opt_ctrl.append(predicted_wellbeing_change_opt_ctrl)

        input_brute_force = strats.brute_force_strategy(locf_X, U, target_state, admissible_inputs, brute_force_time_horizon, rho, online=True)
        predictions_brute_force = predict_n_steps_lds(locf_X, U, index, input_brute_force, n_steps)
        predicted_wellbeing_change_brute_force = np.mean(predictions_brute_force, axis=0) - locf_X[index]
        wellbeing_changes_brute_force.append(predicted_wellbeing_change_brute_force)

        input_max_ac = strats.max_ac_strategy(locf_X, U, admissible_inputs, online=True)
        predictions_max_ac = predict_n_steps_lds(locf_X, U, index, input_max_ac, n_steps)
        predicted_wellbeing_change_max_ac = np.mean(predictions_max_ac, axis=0) - locf_X[index]
        wellbeing_changes_max_ac.append(predicted_wellbeing_change_max_ac)


    real_wellbeings_changes_array = np.array(real_wellbeing_changes)
    real_wellbeing_changes_mean = np.mean(real_wellbeings_changes_array)

    wellbeing_changes_opt_ctrl_array = np.array(wellbeing_changes_opt_ctrl)
    wellbeing_changes_opt_ctrl_mean = np.mean(wellbeing_changes_opt_ctrl_array)

    wellbeing_changes_brute_force_array = np.array(wellbeing_changes_brute_force)
    wellbeing_changes_brute_force_mean = np.mean(wellbeing_changes_brute_force_array)

    wellbeing_changes_max_ac_array = np.array(wellbeing_changes_max_ac)
    wellbeing_changes_max_ac_mean = np.mean(wellbeing_changes_max_ac_array)
    
    difference_wellbeing_opt_ctrl = wellbeing_changes_opt_ctrl_mean - real_wellbeing_changes_mean
    difference_wellbeing_brute_force = wellbeing_changes_brute_force_mean - real_wellbeing_changes_mean
    difference_wellbeing_max_ac = wellbeing_changes_max_ac_mean - real_wellbeing_changes_mean
    print(f'opt_ctrl difference: {difference_wellbeing_opt_ctrl}')
    print(f'brute_force difference: {difference_wellbeing_brute_force}')
    print(f'max_ac difference: {difference_wellbeing_max_ac}')
    print()
    if np.isnan(difference_wellbeing_opt_ctrl):
        count_nan +=1
        continue

    overall_wellbeing_change_opt_ctrl.append(difference_wellbeing_opt_ctrl)
    overall_wellbeing_change_brute_force.append(difference_wellbeing_brute_force)
    overall_wellbeing_change_max_ac.append(difference_wellbeing_max_ac)

print("--------------------------------------")
print(f'nan count = {count_nan}')

mean_difference_opt_ctrl, std_error_difference_opt_ctrl = mean_and_standard_error(overall_wellbeing_change_opt_ctrl)
# Convert results to a DataFrame to save them to a csv file
df_opt_ctrl = pd.DataFrame({"mean": [mean_difference_opt_ctrl], "std_error": [std_error_difference_opt_ctrl]})
filepath1 = os.path.join("results_control_strategies", "opt_ctrl_lds")
df_opt_ctrl.to_csv(filepath1, index=False)

mean_difference_brute_force, std_error_difference_brute_force = mean_and_standard_error(overall_wellbeing_change_brute_force)
# Convert results to a DataFrame to save them to a csv file
df_brute_force = pd.DataFrame({"mean": [mean_difference_brute_force], "std_error": [std_error_difference_brute_force]})
filepath2 = os.path.join("results_control_strategies", "brute_force_lds")
df_brute_force.to_csv(filepath2, index=False)

mean_difference_max_ac, std_error_difference_max_ac = mean_and_standard_error(overall_wellbeing_change_max_ac)
# Convert results to a DataFrame to save them to a csv file
df_max_ac = pd.DataFrame({"mean": [mean_difference_max_ac], "std_error": [std_error_difference_max_ac]})
filepath3 = os.path.join("results_control_strategies", "max_ac_lds")
df_max_ac.to_csv(filepath3, index=False)


strategies = ["Opt Ctrl", "Brute Force", "Max AC"]
means = [mean_difference_opt_ctrl, mean_difference_brute_force, mean_difference_max_ac]
std_errors = [std_error_difference_opt_ctrl, std_error_difference_brute_force, std_error_difference_max_ac]

# Create bar plot
x = np.arange(len(strategies))  # X-axis positions
plt.figure(figsize=(8, 6))
plt.bar(x, means, yerr=std_errors, capsize=5, color=["blue", "green", "red"], alpha=0.7)

# Labels and title
plt.xticks(x, strategies)
plt.ylabel("Mean Wellbeing Change")
plt.title("Comparison of Control Strategies")
plt.grid(axis="y", linestyle="--", alpha=0.6)

# Show plot
plt.show()