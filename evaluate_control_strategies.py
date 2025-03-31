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
subfolders = ["MRT1/processed_csv_no_con", "MRT2/processed_csv_no_con", "MRT3/processed_csv_no_con"]

online = False
n_steps = [1,2,3,4,5,6,7,8,9,10,11,12]
brute_force_time_horizon = 5
rho = 1

dataset_list_lds, files = utils.load_dataset(data_folder, subfolders, emas, emis, centered=True, exclude_constant_columns=False)
dataset_list_rnn, _ = utils.load_dataset(data_folder, subfolders, emas, emis, centered=False, exclude_constant_columns=False)

def get_valid_ratio(data):
    # Valid rows are rows without nan value where also the next row has no nan value
    valid = ~np.isnan(data).any(axis=1)
    valid_rows = valid[:-1] & valid[1:]
    total = valid_rows.sum()
    return total/len(data)

def locf(X_train): 
    df_helper_locf = pd.DataFrame(X_train).copy()
    df_helper_locf.ffill(inplace=True)
    X_train_locf = df_helper_locf.to_numpy()
    return X_train_locf

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

def find_model_path(file, index=None):
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
    if index is not None:
        # Online scenario: Choose model that is trained up to index
        valid_keys = [k for k in model_paths_dict.keys() if k <= index]
        if not valid_keys:
            return False
        key_online = max(valid_keys)  # Get the closest model up to `index`
        model_path_scenario = model_paths_dict[key_online] 
    else:
        # Offline scenario: Choose the last trained model
        max_key = max(model_paths_dict.keys())
        model_path_scenario = model_paths_dict[max_key] 
    return model_path_scenario

def predict_n_step_rnn(index_row, control_input, n_step, model_path):
    index_row_tensor = tc.from_numpy(index_row).to(dtype=tc.float32)

    # U_strategy is a 2D tensor where the first row is the control input and the rest is filled with zeros
    control_input_tensor = tc.from_numpy(control_input).to(dtype=tc.float32)
    U_strategy = tc.zeros((n_step, len(emis)), dtype=tc.float32)
    U_strategy[0] = control_input_tensor

    predictions_overall = []
    
    for i in range(10):
        model_nr = str(i+1).zfill(3)
        model_path_specific = os.path.join(model_path, model_nr)
        try:
            model = PLRNN(load_model_path=model_path_specific)
        except AssertionError as e:
            print(f"Error: {e}. No model found at {model_path_specific}. Exiting function.")
            return []
        predictions = model.generate_free_trajectory(index_row_tensor, n_step, inputs = U_strategy, prewarm_data=None, prewarm_inputs=None)

        predictions_overall.append(predictions[n_step-1])

    predictions_stacked = tc.stack(predictions_overall)
    # Compute the mean across across the 10 models
    predictions_mean = predictions_stacked.mean(dim=0)

    predictions_numpy = predictions_mean.numpy()
    #predictions_numpy[constant_columns] = np.nan # Filter out predictions for constant columns
    return predictions_numpy

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
overall_wellbeing_change_real_emi = {step: [] for step in n_steps} # -> bias of the predictions with the rnn
overall_wellbeing_change_no_emi = {step: [] for step in n_steps}

prediction_bias_rnn_list = [] # Append (prediction - real) every time a prediction is made -> Each prediction (with non nan real target) has the same influence on the bias, so that it doesn't depend on number of interventions and missing data of participants

count_nan = 0

#files_ratio80_split85 = ['data\\MRT1/processed_csv_no_con\\11228_28.csv', 'data\\MRT1/processed_csv_no_con\\11228_34.csv', 'data\\MRT1/processed_csv_no_con\\11228_35.csv', 'data\\MRT1/processed_csv_no_con\\11228_52.csv', 'data\\MRT2/processed_csv_no_con\\12600_25.csv', 'data\\MRT2/processed_csv_no_con\\12600_30.csv', 'data\\MRT2/processed_csv_no_con\\12600_32.csv', 'data\\MRT3/processed_csv_no_con\\12600_221.csv', 'data\\MRT3/processed_csv_no_con\\12600_239.csv', 'data\\MRT3/processed_csv_no_con\\12600_241.csv']

for idx, (dataset, dataset_rnn) in enumerate(zip(dataset_list_lds, dataset_list_rnn)):
    print(files[idx])

    model_path_rnn = find_model_path(files[idx]) # Gets overwritten in the online scenario
    if model_path_rnn == False:
        # For 33 participants there is no trained model
        print("No model found\n")
        continue

    # Mot anymore? Exclude constant columns: used to compute the control strategies (infer A and B on X, U)

    X, U = dataset['X'], dataset['Inp'] # Centered
    # Keep all columns because RNN needs shape (.., 15) to make predictions
    X_rnn = dataset_rnn['X']

    mean_emas = np.nanmean(X_rnn)
    if mean_emas > 5.079: # 50-th lowest mean
        print("Participant is too happy ):")
        continue

    #valid_ratio = get_valid_ratio(X)
    #if valid_ratio < 0.8:
    #    continue      
    
    n_items = X.shape[1]
    n_inputs = U.shape[1]
    target_state = np.full(n_items, -3)
    admissible_inputs = np.eye(n_inputs)

    input_rows_indices = np.where(~np.all(U == 0, axis=1))[0].tolist()

    wellbeing_differences_opt_ctrl = {step: [] for step in n_steps}
    wellbeing_differences_brute_force = {step: [] for step in n_steps}
    wellbeing_differences_max_ac = {step: [] for step in n_steps}
    wellbeing_differences_real_emi = {step: [] for step in n_steps}
    wellbeing_differences_no_emi = {step: [] for step in n_steps}

    locf_X = locf(X) # In X constant columns are excluded
    locf_X_rnn = locf(X_rnn)

    # To exclude constant columns in the predictions
    constant_columns = [i for i in range(locf_X_rnn.shape[1]) if np.unique(locf_X_rnn[:, i]).size == 1]

    if not online:
        input_opt_ctrl_offline = strats.optimal_control_strategy(locf_X, U, target_state, admissible_inputs, rho, online)
        input_brute_force_offline = strats.brute_force_strategy(locf_X, U, target_state, admissible_inputs, brute_force_time_horizon, rho, online)
        input_max_ac_offline = strats.max_ac_strategy(locf_X, U, admissible_inputs, online)

    for index in input_rows_indices:

        if online:
            if index < 80:
                continue
            
            locf_X_online = locf_X[:index, :]
            U_online = U[:index, :]

            input_opt_ctrl = strats.optimal_control_strategy(locf_X_online, U_online, target_state, admissible_inputs, rho, online)
            input_brute_force = strats.brute_force_strategy(locf_X_online, U_online, target_state, admissible_inputs, brute_force_time_horizon, rho, online)
            input_max_ac = strats.max_ac_strategy(locf_X_online, U_online, admissible_inputs, online)

        else:
            # The offline control strategies return 2d arrays the same shape as X
            input_opt_ctrl = input_opt_ctrl_offline[index]
            input_brute_force = input_brute_force_offline[index]
            input_max_ac = input_max_ac_offline[index]

        for step in n_steps:
            if index + step >= len(X_rnn):
                break

            index_row = X_rnn[index]
            if np.isnan(index_row).all():
                break
            
            real_future_step = X_rnn[index + step] # Assume the intervention is done after the emas are answered
            if np.isnan(real_future_step).all():
                continue

            prediction_opt_ctrl = predict_n_step_rnn(index_row, input_opt_ctrl, step, model_path_rnn)
            #print(f'step = {step}, index = {index}')
            #print(index_row)
            #print(prediction_opt_ctrl)
            #print(real_future_step)
            #print()
            wellbeing_differences_opt_ctrl[step].append(prediction_opt_ctrl - real_future_step)

            prediction_brute_force = predict_n_step_rnn(index_row, input_brute_force, step, model_path_rnn)
            wellbeing_differences_brute_force[step].append(np.mean(prediction_brute_force - real_future_step))

            prediction_max_ac = predict_n_step_rnn(index_row, input_max_ac, step, model_path_rnn)
            wellbeing_differences_max_ac[step].append(np.mean(prediction_max_ac - real_future_step))

            prediction_real_emi = predict_n_step_rnn(index_row, U[index], step, model_path_rnn)
            wellbeing_differences_real_emi[step].append(np.mean(prediction_real_emi - real_future_step))

            prediction_no_emi = predict_n_step_rnn(index_row, np.zeros(n_inputs), step, model_path_rnn)
            wellbeing_differences_no_emi[step].append(np.mean(prediction_no_emi - real_future_step))

            prediction_bias_rnn_list.append(prediction_real_emi - real_future_step)

    for step in n_steps:
        
        wellbeing_differences_opt_ctrl_mean = np.mean(wellbeing_differences_opt_ctrl[step])
        wellbeing_differences_brute_force_mean = np.mean(wellbeing_differences_brute_force[step])
        wellbeing_differences_max_ac_mean = np.mean(wellbeing_differences_max_ac[step])
        wellbeing_differences_real_emi_mean = np.mean(wellbeing_differences_real_emi[step])
        wellbeing_differences_no_emi_mean = np.mean(wellbeing_differences_no_emi[step])

        print(f'Prediction Length (step): {step}')
        print(f'constant columns: {constant_columns}')
        print(f'opt_ctrl difference: {wellbeing_differences_opt_ctrl_mean}')
        print(f'brute_force difference: {wellbeing_differences_brute_force_mean}')
        print(f'max_ac difference: {wellbeing_differences_max_ac_mean}')
        print(f'real emi difference: {wellbeing_differences_real_emi_mean}')
        print(f'no emi difference: {wellbeing_differences_no_emi_mean}')
        print()

        if step == 1:  # Store filenames only once per dataset
            filenames.append(files[idx])

        overall_wellbeing_change_opt_ctrl[step].append(wellbeing_differences_opt_ctrl_mean)
        overall_wellbeing_change_brute_force[step].append(wellbeing_differences_brute_force_mean)
        overall_wellbeing_change_max_ac[step].append(wellbeing_differences_max_ac_mean)
        overall_wellbeing_change_real_emi[step].append(wellbeing_differences_real_emi_mean)
        overall_wellbeing_change_no_emi[step].append(wellbeing_differences_no_emi_mean)
    print(40*"-")

extracted_filenames = []
for file in filenames:
    mrt_number = file.split('\\')[1].split('/')[0]
    file_name = os.path.basename(file).split('.')[0] 
    extracted_part = f"{mrt_number}-{file_name}"
    extracted_filenames.append(extracted_part)


# Convert results into DataFrame format
data = {"file": extracted_filenames}
for step in n_steps:
    data[f"optimal control (n={step})"] = overall_wellbeing_change_opt_ctrl.get(step, [])
    data[f"brute force (n={step})"] = overall_wellbeing_change_brute_force.get(step, [])
    data[f"max ac (n={step})"] = overall_wellbeing_change_max_ac.get(step, [])
    data[f"real emi (n={step})"] = overall_wellbeing_change_real_emi.get(step, [])
    data[f"no emi (n={step})"] = overall_wellbeing_change_no_emi.get(step, [])

prediction_bias_rnn_array = np.array(prediction_bias_rnn_list)
prediction_bias_rnn = np.nanmean(prediction_bias_rnn_array)
print(f'Prediction bias rnn: {prediction_bias_rnn}')

df_results = pd.DataFrame(data)
df_results["prediction bias"] = prediction_bias_rnn
filepath1 = os.path.join("results_control_strategies", "results_rnn_offline_step_lowest50.csv")
df_results.to_csv(filepath1, index=False)
print(f"Results saved to {filepath1}")

# Questions to this approach:
# Should you use the latest model (greatest now) for the predictions -> Yes?
# Is the intervention done after the emas are answered in a time step
# Should you include interventions in nan-rows? I exclude them because this row serves as input for the rnn to make the predictions

# Should every participant contribute equally to the mean wellbeing difference per strategy?
#  -> should the mean be taken over the wellbeing difference per intervention rather than participant?

# Maybe test if the model has the tendency to predict lower values (how?) -> by comparing it with the prediction for no-emi

# Online scenario: should U be updated with the control input to compute further control inputs
