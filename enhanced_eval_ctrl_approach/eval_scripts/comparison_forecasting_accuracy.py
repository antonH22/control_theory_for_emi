import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from enhanced_eval_ctrl_approach import myutils

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

### Comparison of RNN and LDS forecasting performance (5.1.5, Figure 5)

def prediction_errors_per_participant(participant_nr, folder_path_model, data_directory, step_by_step=False, n_steps=12):
    " Computes n-step ahead prediction MAEs (step_by_step == False) for both RNN and LDS models for a single participant. "
    mae_per_step_overall_list_rnn = []
    mae_per_step_overall_list_lds = []

    X, U = myutils.load_data(participant_nr, data_directory, centered=False)
    X_centered, _ = myutils.load_data(participant_nr, data_directory, centered=True)

    model_paths = myutils.get_model_paths(participant_nr, folder_path_model)
    for now, model_path in model_paths.items():
        mae_per_step_list_rnn = myutils.prediction_error_rnn(model_path, now, step_by_step, n_steps, X, U)
        mae_per_step_list_lds = myutils.prediction_error_lds(now, step_by_step, n_steps, X_centered, U)
        if len(mae_per_step_list_rnn) == len(mae_per_step_list_lds):
            mae_per_step_overall_list_rnn.append(mae_per_step_list_rnn)
            mae_per_step_overall_list_lds.append(mae_per_step_list_lds)

    mae_per_step_overall_array_rnn = np.array(mae_per_step_overall_list_rnn)
    mae_mean_per_step_array_rnn = np.nanmean(mae_per_step_overall_array_rnn, axis = 0)

    mae_per_step_overall_array_lds = np.array(mae_per_step_overall_list_lds)
    mae_mean_per_step_array_lds = np.nanmean(mae_per_step_overall_array_lds, axis = 0)

    return mae_mean_per_step_array_rnn, mae_mean_per_step_array_lds

rnn_model_path_MRT1 = "D:/v2_MRT1_every_valid_day"
rnn_model_path_MRT2 = "D:/v2_MRT2_every_valid_day"
rnn_model_path_MRT3 = "D:/v2_MRT3_every_valid_day"

data_folder_MRT1 = "data/MRT1/processed_csv_no_con"
data_folder_MRT2 = "data/MRT2/processed_csv_no_con"
data_folder_MRT3 = "data/MRT3/processed_csv_no_con"

step_by_step = False # False: Model uses EMA-values at time t to predict step t+1; then uses its own predictions for subsequent steps
n_steps = 12

def process_participants_per_mrt(participants, folder_path, data_folder):
    " Computes and stores RNN and LDS prediction errors for all participants of one MRT. "
    global num_iterations
    for participant in participants:
        mae_n_mean_array_rnn, mae_n_mean_array_lds = prediction_errors_per_participant(
            participant, folder_path, data_folder, step_by_step, n_steps)
        mae_n_overall_list_rnn.append(mae_n_mean_array_rnn)
        mae_n_overall_list_lds.append(mae_n_mean_array_lds)
        num_iterations += 1
        print(f'Iteration {num_iterations}/143')

# Initialization
participants_MRT1 = myutils.extract_participant_ids(rnn_model_path_MRT1)
participants_MRT2 = myutils.extract_participant_ids(rnn_model_path_MRT2)
participants_MRT3 = myutils.extract_participant_ids(rnn_model_path_MRT3)

num_iterations = 0
mae_n_overall_list_rnn = []
mae_n_overall_list_lds = []

process_participants_per_mrt(participants_MRT1, rnn_model_path_MRT1, data_folder_MRT1)
process_participants_per_mrt(participants_MRT2, rnn_model_path_MRT2, data_folder_MRT2)
process_participants_per_mrt(participants_MRT3, rnn_model_path_MRT3, data_folder_MRT3)

# Save the rnn results to a csv file
mae_n_overall_array_rnn = np.array(mae_n_overall_list_rnn)
output_directory = "results_replicated"
output_file_name = "mae_rnn_forecast.csv"
df = pd.DataFrame(mae_n_overall_array_rnn)
output_file_path = os.path.join(output_directory, output_file_name)
df.to_csv(output_file_path, index=False, header=False)
print(f"{output_file_name} saved to {output_file_path}")

# Save the lds results to a csv file
mae_n_overall_array_lds = np.array(mae_n_overall_list_lds)
output_file_name = "mae_lds_forecast.csv"
df = pd.DataFrame(mae_n_overall_array_lds)
output_file_path = os.path.join(output_directory, output_file_name)
df.to_csv(output_file_path, index=False, header=False)
print(f"{output_file_name} saved to {output_file_path}")