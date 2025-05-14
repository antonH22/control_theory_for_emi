import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from enhanced_eval_ctrl_approach import myutils

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

### Comparison of RNN and LDS prediction accuracy across empirical valid ratios and training set lengths (5.1.5, Figure 6)

folder_path_MRT1 = "D:/v2_MRT1_every_valid_day"
folder_path_MRT2 = "D:/v2_MRT2_every_valid_day"
folder_path_MRT3 = "D:/v2_MRT3_every_valid_day"

data_folder_MRT1 = "data/MRT1/processed_csv_no_con"
data_folder_MRT2 = "data/MRT2/processed_csv_no_con"
data_folder_MRT3 = "data/MRT3/processed_csv_no_con"

participants_MRT1 = myutils.extract_participant_ids(folder_path_MRT1)
participants_MRT2 = myutils.extract_participant_ids(folder_path_MRT2)
participants_MRT3 = myutils.extract_participant_ids(folder_path_MRT3)

step_by_step = True # True: Model predicts each next step using the observed EMA values at the current time step
n_steps = 12

valid_ratio_per_now = []
train_set_length_per_now = []
mae_per_now_rnn = []
mae_per_now_lds = []

num_iterations = 0
def process_participants_per_mrt(participants, folder_path, data_folder):
    """ Computes and stores (averaged step-by-step) prediction MAEs for RNN and LDS models, 
    with corresponding valid ratio and training set length for all participants of one MRT. """
    global num_iterations
    for participant in participants:
        model_paths = myutils.get_model_paths(participant, folder_path)
        X, U = myutils.load_data(participant, data_folder, centered=False)
        X_centered, _ = myutils.load_data(participant, data_folder, centered=True)
        for now, model_path in model_paths.items():
            mae_list_rnn = myutils.prediction_error_rnn(model_path, now, step_by_step, n_steps, X, U)
            mae_list_lds = myutils.prediction_error_lds(now, step_by_step, n_steps, X_centered, U)
            if np.isnan(mae_list_rnn).all() or np.isnan(mae_list_lds).all():
                continue
            X_test = X[:now]
            valid_ratio = myutils.get_valid_ratio(X_test)
            valid_ratio_per_now.append(valid_ratio)
            train_set_length_per_now.append(now)
            mae_per_now_rnn.append(np.nanmean(mae_list_rnn))
            mae_per_now_lds.append(np.nanmean(mae_list_lds))
        num_iterations += 1
        print(f'Iteration {num_iterations}/143')

process_participants_per_mrt(participants_MRT1, folder_path_MRT1, data_folder_MRT1)
process_participants_per_mrt(participants_MRT2, folder_path_MRT2, data_folder_MRT2)
process_participants_per_mrt(participants_MRT3, folder_path_MRT3, data_folder_MRT3)

# Convert results to DataFrame to save it to a csv file
df_rnn = pd.DataFrame({
    "ratio": valid_ratio_per_now,
    "trainlen": train_set_length_per_now,
    "mae_rnn": mae_per_now_rnn,
    "mae_lds": mae_per_now_lds
})
filename = f'mae_vs_ratio_trainlen.csv'
filepath = os.path.join("results_replicated", filename)
df_rnn.to_csv(filepath, index=False)
print(f'Final rnn results saved to {filename}')