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

n_steps = [1,2,3,4,5,6,7,8,9,10,11,12]
brute_force_time_horizon = 5
rho = 1

dataset_list, files = utils.load_dataset(data_folder, subfolders, emas, emis, centered=False, exclude_constant_columns=True)
dataset_list_rnn, _ = utils.load_dataset(data_folder, subfolders, emas, emis, centered=False, exclude_constant_columns=False)

def locf(X_train): 
    df_helper_locf = pd.DataFrame(X_train).copy()
    df_helper_locf.ffill(inplace=True)
    X_train_locf = df_helper_locf.to_numpy()
    return X_train_locf

def compute_real_future_mean(data, index, n_steps):
    future_steps = data[index + 1 : index + n_steps + 1]
    if index + n_steps >= data.shape[0] or np.isnan(future_steps).all():
        return False
    future_mean = np.nanmean(future_steps, axis=0)
    return future_mean

def compute_predictions_mean(predictions, constant_columns):
    if constant_columns:
        # Exclude constant columns
        filtered_predictions = np.delete(predictions, constant_columns, axis=1)
        return np.mean(filtered_predictions, axis=0)
    else:
        return np.mean(predictions, axis=0)

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
overall_wellbeing_change_opt_ctrl = {step: [] for step in n_steps}
overall_wellbeing_change_brute_force = {step: [] for step in n_steps}
overall_wellbeing_change_max_ac = {step: [] for step in n_steps}


count_nan = 0

for idx, (dataset, dataset_rnn) in enumerate(zip(dataset_list, dataset_list_rnn)):
    model_path_rnn = find_model_path(files[idx])

    X, U = dataset['X'], dataset['Inp']
    X_rnn = dataset_rnn['X']
    
    if len(X) < num_rows_threshold:
        continue

    
    print(files[idx])
    if model_path_rnn == False:
        print('Failed to get model path')
        print()
        continue
    
    n_items = X.shape[1]
    n_inputs = U.shape[1]
    target_state = np.full(n_items, -3)
    admissible_inputs = np.eye(n_inputs)

    input_rows_indices = np.where(~np.all(U == 0, axis=1))[0].tolist()

    real_future_means = {step: [] for step in n_steps}
    predicted_means_opt_ctrl = {step: [] for step in n_steps}
    predicted_means_brute_force = {step: [] for step in n_steps}
    predicted_means_max_ac = {step: [] for step in n_steps}


    for index in input_rows_indices:
        if index < 20:
            continue

        locf_X = locf(X)
        locf_X_rnn = locf(X_rnn)

        constant_columns = [i for i in range(locf_X_rnn.shape[1]) if np.unique(locf_X_rnn[:, i]).size == 1]

        input_opt_ctrl = strats.optimal_control_strategy(locf_X, U, target_state, admissible_inputs, rho, online=True)
        input_brute_force = strats.brute_force_strategy(locf_X, U, target_state, admissible_inputs, brute_force_time_horizon, rho, online=True)
        input_max_ac = strats.max_ac_strategy(locf_X, U, admissible_inputs, online=True)

        for step in n_steps:
            real_future_mean = compute_real_future_mean(X_rnn, index, step)
            if real_future_mean is False:
                continue
            real_future_means[step].append(real_future_mean)

            predictions_opt_ctrl = predict_n_steps_rnn(locf_X_rnn, U, index, input_opt_ctrl, step, model_path_rnn)
            predicted_means_opt_ctrl[step].append(compute_predictions_mean(predictions_opt_ctrl, constant_columns))

            predictions_brute_force = predict_n_steps_rnn(locf_X_rnn, U, index, input_brute_force, step, model_path_rnn)
            predicted_means_brute_force[step].append(compute_predictions_mean(predictions_brute_force, constant_columns))

            predictions_max_ac = predict_n_steps_rnn(locf_X_rnn, U, index, input_max_ac, step, model_path_rnn)
            predicted_means_max_ac[step].append(compute_predictions_mean(predictions_max_ac, constant_columns))

    for step in n_steps:
        if not real_future_means[step]:  # Skip if no data collected
            continue
        
        real_mean = np.mean(real_future_means[step])
        predicted_opt_ctrl_mean = np.mean(predicted_means_opt_ctrl[step])
        predicted_brute_force_mean = np.mean(predicted_means_brute_force[step])
        predicted_max_ac_mean = np.mean(predicted_means_max_ac[step])

        difference_wellbeing_opt_ctrl = predicted_opt_ctrl_mean - real_mean
        difference_wellbeing_brute_force = predicted_brute_force_mean - real_mean
        difference_wellbeing_max_ac = predicted_max_ac_mean - real_mean

        print(f'Prediction Length (step): {step}')
        print(f'constant columns: {constant_columns}')
        print(f'opt_ctrl difference: {difference_wellbeing_opt_ctrl}')
        print(f'brute_force difference: {difference_wellbeing_brute_force}')
        print(f'max_ac difference: {difference_wellbeing_max_ac}')
        print()

        if np.isnan(difference_wellbeing_opt_ctrl):
            count_nan += 1
            continue

        if step == 1:  # Store filenames only once per dataset
            filenames.append(files[idx])

        overall_wellbeing_change_opt_ctrl[step].append(difference_wellbeing_opt_ctrl)
        overall_wellbeing_change_brute_force[step].append(difference_wellbeing_brute_force)
        overall_wellbeing_change_max_ac[step].append(difference_wellbeing_max_ac)
    print(40*"-")

extracted_filenames = []
for file in filenames:
    mrt_number = file.split('\\')[1].split('/')[0]
    file_name = os.path.basename(file).split('.')[0] 
    extracted_part = f"{mrt_number}-{file_name}"
    extracted_filenames.append(extracted_part)

print(f'nan count = {count_nan}')

# Convert results into DataFrame format
data = {"file": extracted_filenames}
for step in n_steps:
    data[f"optimal control (n={step})"] = overall_wellbeing_change_opt_ctrl.get(step, [])
    data[f"brute force (n={step})"] = overall_wellbeing_change_brute_force.get(step, [])
    data[f"max ac (n={step})"] = overall_wellbeing_change_max_ac.get(step, [])

df_results = pd.DataFrame(data)
filepath1 = os.path.join("results_control_strategies", "results_rnn_steps_mean.csv")
df_results.to_csv(filepath1, index=False)
print(f"Results saved to {filepath1}")

# Questions to this approach:
# Should you use the latest model (greatest now) for the predictions?
# Should you include interventions in nan-rows?

# Should you compute the difference of the means of the next n_steps or only for the steps that arent nan in the real data (second is better because you dont make assumption that they would answer mean)
# Should every file contribute equally to the mean wellbeing difference per strategy?
#  -> should the mean be taken over the wellbeing difference per intervention rather than participant?

# Maybe test if the model has the tendency to predict lower values (how?)
