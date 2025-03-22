from ctrl import utils
from ctrl import discrete_optimal_control as doc
from ctrl import control_strategies as strats
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import re
import torch as tc

import os

from bptt.plrnn import PLRNN

rnn_model_path_MRT1 = "D:/v2_MRT1_every_valid_day"
rnn_model_path_MRT2 = "D:/v2_MRT2_every_valid_day"
rnn_model_path_MRT3 = "D:/v2_MRT3_every_valid_day"

emas = ['EMA_mood', 'EMA_disappointed', 'EMA_scared', 'EMA_worry', 'EMA_down', 'EMA_sad', 'EMA_confidence', 'EMA_stress', 'EMA_lonely', 'EMA_energetic', 'EMA_concentration', 'EMA_resilience', 'EMA_tired', 'EMA_satisfied', 'EMA_relaxed']
emis = ['interactive1', 'interactive2', 'interactive3', 'interactive4', 'interactive5', 'interactive6', 'interactive7', 'interactive8']

data_folder = "data"
subfolders = ["MRT1/processed_csv_no_con","MRT2/processed_csv_no_con","MRT3/processed_csv_no_con"]

num_rows_threshold = 50 # One file is excluded

n_steps = 10
brute_force_time_horizon = 5
rho = 1

dataset_list, files = utils.load_dataset(data_folder, subfolders, emas, emis, centered=False, exclude_constant_columns=True)

def locf(X_train): 
    df_helper_locf = pd.DataFrame(X_train).copy()
    df_helper_locf.ffill(inplace=True)
    X_train_locf = df_helper_locf.to_numpy()
    return X_train_locf

def compute_real_wellbeing_change(data, index, n_steps):
    future_steps = data[index + 1 : index + n_steps + 1]
    if index + n_steps >= data.shape[0] or np.isnan(future_steps).all():
        return False
    future_mean = np.nanmean(future_steps, axis=0)
    wellbeing_change = future_mean - data[index]
    return wellbeing_change

def predict_n_steps_lds(locf_X, U, index, control_input, n_steps):
    A, B, lmbda = utils.stable_ridge_regression(locf_X, U)
    predictions = []

    U_strategy = U.copy()
    U_strategy[index] = control_input
    
    for i in range(n_steps):
        x_next = doc.step(A, B, locf_X[index + i], U_strategy[index + i])
        predictions.append(x_next)
    return predictions

def get_model_paths(participant_nr, folder_path):
    model_paths_dict = {}
    for filename in os.listdir(folder_path):
        match = re.search(rf'data_\d+_\d+\.csv_participant_{participant_nr}_date_(\d+\.\d+)', filename)
        if match:
            timestep = float(match.group(1))
            model_paths_dict[timestep] = os.path.join(folder_path, filename)
    return model_paths_dict

def find_model_path(file):
    # Get MRT number
    match = re.search(r"MRT(\d)", file)
    mrt_nr = int(match.group(1))
    model_paths = {1: rnn_model_path_MRT1, 2: rnn_model_path_MRT2, 3: rnn_model_path_MRT3}
    model_path = model_paths.get(mrt_nr)
    # Get participant number
    match = re.search(r'_(\d+)\.csv$', file)
    participant_nr = int(match.group(1))
    model_paths_dict = get_model_paths(participant_nr, model_path)
    if not model_paths_dict:
        return False
    # Choose the last trained model
    max_key = max(model_paths_dict.keys())
    last_model_path = model_paths_dict[max_key]
    return last_model_path

def predict_n_steps_rnn(locf_X, U, index, control_input, n_steps, model_path):
    locf_X_tensor = tc.from_numpy(locf_X).to(dtype=tc.float32)
    U_tensor = tc.from_numpy(U).to(dtype=tc.float32)
    control_input_tensor = tc.from_numpy(control_input).to(dtype=tc.float32)
    U_strategy = U_tensor.clone()
    U_strategy[index] = control_input_tensor
    predictions_overall = []
    for i in range(10):
        model_nr = str(i+1).zfill(3)
        model_path_specific = os.path.join(model_path, model_nr)
        try:
            model = PLRNN(load_model_path=model_path_specific)
        except AssertionError as e:
            print(f"Error: {e}. No model found at {model_path_specific}. Exiting function.")
            return []
        predictions = model.generate_free_trajectory(locf_X_tensor[index], n_steps, inputs = U_strategy[index:index+n_steps], prewarm_data=None, prewarm_inputs=None)
        predictions_overall.append(predictions)
    predictions_stacked = tc.stack(predictions_overall)
    # Compute the mean across across the 10 models  
    predictions_mean = tc.mean(predictions_stacked, dim=0)
    predictions_mean_numpy = predictions_mean.numpy()
    return predictions_mean_numpy

def mean_and_standard_error(data):
    data = np.array(data)
    mean_value = np.mean(data)
    std_dev = np.std(data)
    n = len(data)
    standard_error = std_dev / np.sqrt(n)
    return mean_value, standard_error

filenames = []
overall_real_wellbeing_change = []
overall_wellbeing_change_opt_ctrl = []
overall_wellbeing_change_brute_force = []
overall_wellbeing_change_max_ac = []

count_nan = 0
for idx, dataset in enumerate(dataset_list):
    model_path_rnn = find_model_path(files[idx])
    if model_path_rnn == False:
        continue

    X, U = dataset['X'], dataset['Inp']
    if len(X) < num_rows_threshold:
        continue
    
    n_items = X.shape[1]
    n_inputs = U.shape[1]
    target_state = np.full(n_items, -3)
    admissible_inputs = np.eye(n_inputs)

    input_rows_indices = np.where(~np.all(U == 0, axis=1))[0].tolist()
    print(files[idx])

    real_wellbeing_changes = []
    wellbeing_changes_opt_ctrl = []
    wellbeing_changes_brute_force = []
    wellbeing_changes_max_ac = []
    for index in input_rows_indices:
        if index < 20:
            continue
        row_with_intervention = X[index]
        if np.isnan(row_with_intervention).all():
            continue
        real_wellbeing_change = compute_real_wellbeing_change(X, index, n_steps)
        if real_wellbeing_change is False:
            continue  # Skip this index if compute real wellbeing change failed
        real_wellbeing_changes.append(real_wellbeing_change)
        
        locf_X = locf(X)

        input_opt_ctrl = strats.optimal_control_strategy(locf_X, U, target_state, admissible_inputs, rho, online=True)
        predictions_opt_ctrl = predict_n_steps_rnn(locf_X, U, index, input_opt_ctrl, n_steps, model_path_rnn)
        predicted_wellbeing_change_opt_ctrl = np.mean(predictions_opt_ctrl, axis=0) - row_with_intervention
        wellbeing_changes_opt_ctrl.append(predicted_wellbeing_change_opt_ctrl)

        input_brute_force = strats.brute_force_strategy(locf_X, U, target_state, admissible_inputs, brute_force_time_horizon, rho, online=True)
        predictions_brute_force = predict_n_steps_rnn(locf_X, U, index, input_brute_force, n_steps, model_path_rnn)
        predicted_wellbeing_change_brute_force = np.mean(predictions_brute_force, axis=0) - row_with_intervention
        wellbeing_changes_brute_force.append(predicted_wellbeing_change_brute_force)

        input_max_ac = strats.max_ac_strategy(locf_X, U, admissible_inputs, online=True)
        predictions_max_ac = predict_n_steps_rnn(locf_X, U, index, input_max_ac, n_steps, model_path_rnn)
        predicted_wellbeing_change_max_ac = np.mean(predictions_max_ac, axis=0) - row_with_intervention
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

    filenames.append(files[idx])
    print(f'Loop {len(filenames)}')
    overall_wellbeing_change_opt_ctrl.append(difference_wellbeing_opt_ctrl)
    overall_wellbeing_change_brute_force.append(difference_wellbeing_brute_force)
    overall_wellbeing_change_max_ac.append(difference_wellbeing_max_ac)

extracted_filenames = []
for file in filenames:
    # Extract the required part (e.g. MRT1-11228_28)
    mrt_number = file.split('\\')[1].split('/')[0]
    file_name = os.path.basename(file).split('.')[0] 
    extracted_part = f"{mrt_number}-{file_name}"
    extracted_filenames.append(extracted_part)

print("--------------------------------------")
print(f'nan count = {count_nan}')

df_results = pd.DataFrame({"optimal control": overall_wellbeing_change_opt_ctrl, "brute force": overall_wellbeing_change_brute_force, "max ac":overall_wellbeing_change_max_ac, "file": extracted_filenames})
filepath1 = os.path.join("results_control_strategies", "results_rnn.csv")
df_results.to_csv(filepath1, index=False)