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

def get_valid_ratio(data, now):
    X = data['X']
    emas_training = X[:now]
    valid = ~np.isnan(emas_training).any(axis=1)
    valid_rows = valid[:-1] & valid[1:]
    total = valid_rows.sum()
    return total/(now-1)

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

def locf(X_train): 
    df_helper_locf = pd.DataFrame(X_train).copy()
    df_helper_locf.ffill(inplace=True)
    X_train_locf = df_helper_locf.to_numpy()
    return X_train_locf

def model_prediction(now, dataset, step_by_step, n_steps, filename):
    mae_per_step_list = []
    mae_per_step_n_list = []
    X, U = dataset['X'], dataset['Inp']
    # Split data
    X_train = X[:now]
    U_train = U[:now]
    X_train_locf = locf(X_train)
    X_validation = X[now+1:now+n_steps+1]
    
    A, B, lmbda = utils.stable_ridge_regression(X_train_locf, U_train)
    current_row = now
    predictions_list = []

    for i in range(n_steps):
        if current_row >= len(X):
                break
        if step_by_step:
            x_next = doc.step(A, B, X[current_row], U[current_row])
        else: 
            if i == 0:
                x_next = X[now]
            x_next = doc.step(A, B, x_next, U[current_row])

        predictions_list.append(x_next)
        current_row += 1
    
    predictions_np = np.array(predictions_list)
    # Compute the prediction error
    for i in range(len(X_validation)):
        # Compute the MAE per time step nan values
        mae_per_step_n = np.mean(np.abs(predictions_np[i] - X_validation[i]))
        mae_per_step_n_list.append(mae_per_step_n)
        # Skip if there is a NaN value in the validation data
        if np.isnan(X_validation[i]).any():
            continue
        # Skip if there is a NaN value in the prediction (occurs when doing step by step and current row is nan.)
        if np.isnan(predictions_np[i]).any():
            continue
        mae_per_step = np.mean(np.abs(predictions_np[i] - X_validation[i]))
        if np.isnan(mae_per_step):
            print(filename)
            print("Mae ist nan")
        else:
            mae_per_step_list.append(mae_per_step)
    return mae_per_step_list, mae_per_step_n_list

def prediction_errors_per_participant(participant_nr, rnn_model_path_MRT, data_path, step_by_step, n_steps):
    mae_overall_list = []
    now_list = []
    ratio_list = []
    
    csv_path, filename = get_csv_file_path(participant_nr, data_path)
    timesteps = extract_timesteps(filename, rnn_model_path_MRT)
    dataset = utils.csv_to_dataset(csv_path, emas, emis, centered=True, invert_columns=[], exclude_constant_columns=False, remove_initial_nan=False)

    for now in timesteps:
        mae_per_now, _ = model_prediction(now, dataset, step_by_step, n_steps, filename)
        if mae_per_now != []:
            mae_overall_list.append(sum(mae_per_now)/len(mae_per_now))
            #mae_overall_list.extend(mae_per_now)
            now_list.append(now)
            #now_list.extend([now] * len(mae_per_now))
            ratio = get_valid_ratio(dataset, now)
            #ratio_list.extend([ratio]*len(mae_per_now))
            ratio_list.append(ratio)

    return mae_overall_list, now_list, ratio_list


if __name__=='__main__':
    data_folder_MRT1_smoothed = "data/MRT1/processed_csv_no_con_smoothed"
    data_folder_MRT1 = "data/MRT1/processed_csv_no_con"
    data_folder_MRT2_smoothed = "data/MRT2/processed_csv_no_con_smoothed"
    data_folder_MRT2 = "data/MRT2/processed_csv_no_con"
    data_folder_MRT3_smoothed = "data/MRT3/processed_csv_no_con_smoothed"
    data_folder_MRT3 = "data/MRT3/processed_csv_no_con"

    rnn_model_path_MRT1 = "D:/v2_MRT1_every_valid_day"
    rnn_model_path_MRT2 = "D:/v2_MRT2_every_valid_day"
    rnn_model_path_MRT3 = "D:/v2_MRT3_every_valid_day"

    participants_MRT1 = extract_participant_ids(rnn_model_path_MRT1)
    participants_MRT2 = extract_participant_ids(rnn_model_path_MRT2)
    participants_MRT3 = extract_participant_ids(rnn_model_path_MRT3)

    step_by_step = False
    n_steps = 10

    mae_per_participant = []
    valid_ratio_per_participant = []
    train_set_length_per_participant = []
    for participant in participants_MRT1:
        if participant == 52:
            continue
        mae_per_participant_list, train_set_length_list, valid_ratio_list  = prediction_errors_per_participant(participant, rnn_model_path_MRT1, data_folder_MRT1, step_by_step, n_steps)
        mae_per_participant.extend(mae_per_participant_list)
        valid_ratio_per_participant.extend(valid_ratio_list)
        train_set_length_per_participant.extend(train_set_length_list)

    for participant in participants_MRT2:
        if participant == 52 or participant == 64:
            continue
        mae_per_participant_list, train_set_length_list, valid_ratio_list  = prediction_errors_per_participant(participant, rnn_model_path_MRT2, data_folder_MRT2, step_by_step, n_steps)
        mae_per_participant.extend(mae_per_participant_list)
        valid_ratio_per_participant.extend(valid_ratio_list)
        train_set_length_per_participant.extend(train_set_length_list)
    
    for participant in participants_MRT3:
        if participant == 239:
            continue
        mae_per_participant_list, train_set_length_list, valid_ratio_list = prediction_errors_per_participant(participant, rnn_model_path_MRT3, data_folder_MRT3, step_by_step, n_steps)
        mae_per_participant.extend(mae_per_participant_list)
        valid_ratio_per_participant.extend(valid_ratio_list)
        train_set_length_per_participant.extend(train_set_length_list)

    # Convert results to a DataFrame to save them to a csv file
    df_final = pd.DataFrame({
        "ratio": valid_ratio_per_participant,
        "trainlen": train_set_length_per_participant,
        "errors": mae_per_participant,
    })
    # Save to CSV
    filename = f'ratio_trainlen_mae_LDS_n{n_steps}contin_mean.csv'
    filepath = os.path.join("results_ratio_trainlen_compare", filename)
    df_final.to_csv(filepath, index=False)
    print(f'Final results saved to {filename}')
   
    print('Prediction errors LDS')
    print(f'Step-by-step = {step_by_step}')
    print(f'Number of valid predictions: {len(mae_per_participant)}')
    print(f'MAE: {np.mean(mae_per_participant):.3f}')

    plt.scatter(valid_ratio_per_participant, mae_per_participant, s=10, c='blue', marker='o')
    plt.xlabel('valid ratio (real emas)')
    plt.ylabel('mae')
    plt.title('RNN valid ratio vs. mae')
    plt.show()

    plt.scatter(train_set_length_per_participant, mae_per_participant, s=10, c='blue', marker='o')
    plt.xlabel('train set length')
    plt.ylabel('mae')
    plt.title('RNN train set length vs. mae')
    plt.show()

   