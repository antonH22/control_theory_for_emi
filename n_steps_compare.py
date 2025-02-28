from ctrl import discrete_optimal_control as doc
from ctrl import utils
import sys
sys.path.append('..')
import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import re

emas = ['EMA_mood', 'EMA_disappointed', 'EMA_scared', 'EMA_worry', 'EMA_down', 'EMA_sad', 'EMA_confidence', 'EMA_stress', 'EMA_lonely', 'EMA_energetic', 'EMA_concentration', 'EMA_resilience', 'EMA_tired', 'EMA_satisfied', 'EMA_relaxed']
emis = ['interactive1', 'interactive2', 'interactive3', 'interactive4', 'interactive5', 'interactive6', 'interactive7', 'interactive8']

def extract_timesteps(file, model_path):
    filename = os.path.basename(file)
    timestep_pattern = re.compile(rf'{re.escape(filename)}_participant_\d+_date_(\d+)')
    timesteps = []
    
    for file in os.listdir(model_path):
        match = timestep_pattern.search(file)
        if match:
            timesteps.append(int(match.group(1)))
    return sorted(timesteps)

def get_csv_file_path(participant_nr, folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv") and filename.split("_")[1].split(".")[0] == str(participant_nr):
            return os.path.join(folder_path, filename), filename
    return None

def extract_participant_ids(folder_path):
    participant_ids = set()

    for filename in os.listdir(folder_path):
        match = re.search(r'participant_(\d+)', filename)
        if match:
            participant_ids.add(int(match.group(1)))
    
    return sorted(participant_ids)

def model_prediction(now, dataset, step_by_step, n_steps, filename):
    mae_per_step_list = []
    X, U = dataset['X'], dataset['Inp']
    # Split data
    X_train = X[:now]
    U_train = U[:now]
    X_validation = X[now+1:now+n_steps+1]
    
    A, B, lmbda = utils.stable_ridge_regression(X_train, U_train)
    current_row = now
    predictions_list = []

    x_next = X[now]
    for i in range(n_steps):
        if current_row >= len(X):
                break
        if step_by_step:
            x_next = doc.step(A, B, X[current_row], U[current_row])
        else: 
            x_next = doc.step(A, B, x_next, U[current_row])
        predictions_list.append(x_next)
        current_row += 1
    
    predictions_np = np.array(predictions_list)
    # Compute the prediction error
    for i in range(len(X_validation)):
        # Skip if there is a NaN value in the validation data
        if np.isnan(X_validation[i]).any():
            continue
        prediction_np = predictions_np[i]
        if np.isnan(prediction_np).any():
            # Happens when doing step by step and current row is nan.
            continue
        target = X_validation[i]
        mae_per_step = np.mean(np.abs(prediction_np - target))
        if np.isnan(mae_per_step):
            #print(filename)
            #print("Mae ist nan")
            a = []
        else:
            mae_per_step_list.append(mae_per_step)
    return mae_per_step_list

def prediction_errors_per_participant(participant_nr, rnn_model_path_MRT, data_path, step_by_step, n_steps):
    # There is no prediction possible with the rnn
    if participant_nr == 52:
        return []
    mae_overall_list = []
    csv_path, filename = get_csv_file_path(participant_nr, data_path)
    timesteps = extract_timesteps(filename, rnn_model_path_MRT)
    dataset = utils.csv_to_dataset(csv_path, emas, emis, invert_columns=[], remove_initial_nan=False)

    for now in timesteps:
        mae_per_now = model_prediction(now, dataset, step_by_step, n_steps, filename)
        mae_overall_list.extend(mae_per_now)

    return mae_overall_list


if __name__=='__main__':
    data_folder_MRT2_smoothed = "data/MRT2/processed_csv_no_con_smoothed"
    data_folder_MRT2 = "data/MRT2/processed_csv_no_con"
    data_folder_MRT3_smoothed = "data/MRT3/processed_csv_no_con_smoothed"
    data_folder_MRT3 = "data/MRT3/processed_csv_no_con"

    rnn_model_path_MRT2 = "D:/v2_MRT2_every_valid_day"
    rnn_model_path_MRT3 = "D:/v2_MRT3_every_valid_day"

    participants_MRT2 = extract_participant_ids(rnn_model_path_MRT2)
    participants_MRT3 = extract_participant_ids(rnn_model_path_MRT3)

    step_by_step = True
    n_steps = 10

    mae_overall_list = []
    
    for participant in participants_MRT2:
        mae_per_participant_list = prediction_errors_per_participant(participant, rnn_model_path_MRT2, data_folder_MRT2, step_by_step, n_steps)
        mae_overall_list.extend(mae_per_participant_list)
    
    for participant in participants_MRT3:
        mae_per_participant_list = prediction_errors_per_participant(participant, rnn_model_path_MRT3, data_folder_MRT3, step_by_step, n_steps)
        mae_overall_list.extend(mae_per_participant_list)

    print(f'Step-by-step = {step_by_step}')
    print(f'Number of valid predictions: {len(mae_overall_list)}')
    print(f'MAE: {np.mean(mae_overall_list):.4f}')