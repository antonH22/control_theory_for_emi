import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from enhanced_eval_ctrl_approach import myutils
from ctrl import control_strategies as strats
import numpy as np
import pandas as pd
import torch as tc

from bptt.plrnn import PLRNN

### Evaluating control strategies with the RNN model with bias correction (5.1.6, Figure 7) and without bias correction (A.2, Figure 10)
### Evaluating optimal control strategy across empirical valid ratios and training set lengths (A.3, Figure 11, 12)

rnn_model_path_MRT1 = "D:/v2_MRT1_every_valid_day"
rnn_model_path_MRT2 = "D:/v2_MRT2_every_valid_day"
rnn_model_path_MRT3 = "D:/v2_MRT3_every_valid_day"

emas = ['EMA_mood', 'EMA_disappointed', 'EMA_scared', 'EMA_worry', 'EMA_down', 'EMA_sad', 'EMA_confidence', 'EMA_stress', 'EMA_lonely', 'EMA_energetic', 'EMA_concentration', 'EMA_resilience', 'EMA_tired', 'EMA_satisfied', 'EMA_relaxed']
emis = ['interactive1', 'interactive2', 'interactive3', 'interactive4', 'interactive5', 'interactive6', 'interactive7', 'interactive8']

data_folder = "data"
subfolders = ["MRT1/processed_csv_no_con", "MRT2/processed_csv_no_con", "MRT3/processed_csv_no_con"]

save_folder = "results_replicated"
filename_results = "results_control_strategies_offline.csv"
filename_results_data_quality = "_ratio_trainlen.csv"

online = False
bias_correction = False # Bias correction significantly increases runtime
step_x = 1 # Forecasting step for which valid ratios and training set length are extracted (analyzed for )

n_steps = [1,2,3,4,5,6,7,8,9,10,11,12]
brute_force_time_horizon = 5

def compute_bias(X_rnn, U, index, input_rows_indices, step, model_path_rnn):
    " Computes the n-th step prediction bias after interventions before intervention at index_row "
    index_threshold = index - step
    indices_online = [i for i in input_rows_indices if i < index_threshold]
    helper_prediction_error = []
    for index_online in indices_online:
        prediction_real_emi = predict_n_step_rnn(X_rnn[index_online], U[index_online], step, model_path_rnn)
        real_future_step_online = X_rnn[index_online + step]
        if not np.isnan(real_future_step_online).all():
            helper_prediction_error.append(np.mean(prediction_real_emi - real_future_step))
    bias = np.mean(helper_prediction_error)
    return bias

def predict_n_step_rnn(index_row, control_input, n_step, model_path):
    " Predicts the n-th step after intervention (control_input) at index_row. "
    index_row_tensor = tc.from_numpy(index_row).to(dtype=tc.float32)
    # U_strategy is a 2D tensor where the first row is the control input (intervention) and the rest is filled with zeros
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
        # Only returns the prediction n-th step after index_row
        predictions_overall.append(predictions[n_step-1])

    predictions_stacked = tc.stack(predictions_overall)
    # Compute the mean across across the 10 models of each participant
    predictions_mean = predictions_stacked.mean(dim=0)
    predictions_numpy = predictions_mean.numpy()
    return predictions_numpy

dataset_list_lds, files = myutils.load_dataset(data_folder, subfolders, emas, emis, centered=True)
dataset_list_rnn, _ = myutils.load_dataset(data_folder, subfolders, emas, emis, centered=False)

filenames = []
overall_wellbeing_change_opt_ctrl = {step: [] for step in n_steps}
overall_wellbeing_change_brute_force = {step: [] for step in n_steps}
overall_wellbeing_change_max_ac = {step: [] for step in n_steps}
overall_wellbeing_change_real_emi = {step: [] for step in n_steps}
overall_wellbeing_change_no_emi = {step: [] for step in n_steps}
overall_bias_rnn = {step: [] for step in n_steps}

count_nan = 0

# Only for step_x to track the relationship between data quality and well-being difference
ratios_trainset_online = {"valid ratio": [], "trainset length": [], "optimal control": [], "brute force": [], "max ac": [], "real emi": [], "no emi": []}

for dataset, dataset_rnn, file in zip(dataset_list_lds, dataset_list_rnn, files):
    print(file)
    # Check if there is a trained model for this participant
    participant_nr, model_path = myutils.extract_participant_nr_and_model_path(file, rnn_model_path_MRT1, rnn_model_path_MRT2, rnn_model_path_MRT3)
    model_paths_dict = myutils.get_model_paths(participant_nr, model_path)
    if not model_paths_dict:
        # For 33 participants there is no trained model
        print("No model found\n")
        continue

    X, U = dataset['X'], dataset['Inp'] # Centered [-3, 3]
    X_rnn = dataset_rnn['X'] # Non-centered [1, 7]

    num_interventions = np.sum(U)    
    
    n_items = X.shape[1]
    n_inputs = U.shape[1]
    target_state = np.full(n_items, 3)
    admissible_inputs = np.eye(n_inputs)

    input_rows_indices = np.where(~np.all(U == 0, axis=1))[0].tolist()

    wellbeing_differences_opt_ctrl = {step: [] for step in n_steps}
    wellbeing_differences_brute_force = {step: [] for step in n_steps}
    wellbeing_differences_max_ac = {step: [] for step in n_steps}
    wellbeing_differences_real_emi = {step: [] for step in n_steps}
    wellbeing_differences_no_emi = {step: [] for step in n_steps}

    bias_rnn = {step: [] for step in n_steps} # To keep track of the bias correction (bias is corrected on the go)

    locf_X = myutils.locf(X)

    if not online:
        input_opt_ctrl_offline = strats.optimal_control_strategy(locf_X, U, target_state, admissible_inputs, 1, online)
        input_brute_force_offline = strats.brute_force_strategy(locf_X, U, target_state, admissible_inputs, brute_force_time_horizon, 1, online)
        input_max_ac_offline = strats.max_ac_strategy(locf_X, U, admissible_inputs, online)

    for index in input_rows_indices:
        if index < 80:
            # There are no models trained before time step 80
            continue
        # Get the path of the latest available RNN model up to index
        model_path_rnn = myutils.get_latest_model_path(participant_nr, model_path, index)
        if model_path_rnn == False:
            # Sometimes there are no models trained to now <= index. If that's the case continue with next index.
            continue

        index_row = X_rnn[index]
        if np.isnan(index_row).all():
            # Input row is used as a predictor for the RNN and cannot be nan
            continue

        if online:
            locf_X_online = locf_X[:index, :]
            U_online = U[:index, :]

            input_opt_ctrl = strats.optimal_control_strategy(locf_X_online, U_online, target_state, admissible_inputs, 1, online)
            input_brute_force = strats.brute_force_strategy(locf_X_online, U_online, target_state, admissible_inputs, brute_force_time_horizon, 1, online)
            input_max_ac = strats.max_ac_strategy(locf_X_online, U_online, admissible_inputs, online)

        else:
            # The offline control strategies return 2d arrays the same shape as X
            input_opt_ctrl = input_opt_ctrl_offline[index]
            input_brute_force = input_brute_force_offline[index]
            input_max_ac = input_max_ac_offline[index]

        for step in n_steps:
            if index + step >= len(X_rnn):
                break
            
            real_future_step = X_rnn[index + step]
            if np.isnan(real_future_step).all():
                # If there is no real observation you cant compute the wellbeing difference
                continue
            
            if bias_correction:
                 # Bias: Compute the prediction error using the ground truth per step after every intervention before index_row
                bias = compute_bias(X_rnn, U, index, input_rows_indices, step, model_path_rnn)
                bias_rnn[step].append(bias) # To keep track of the bias correction
            else:
                bias = 0
            
            prediction_real_emi = predict_n_step_rnn(index_row, U[index], step, model_path_rnn)
            wellbeing_differences_real_emi[step].append(np.mean(prediction_real_emi - real_future_step) - bias)
            
            prediction_opt_ctrl = predict_n_step_rnn(index_row, input_opt_ctrl, step, model_path_rnn)
            wellbeing_differences_opt_ctrl[step].append(np.mean(prediction_opt_ctrl - real_future_step) - bias)

            prediction_brute_force = predict_n_step_rnn(index_row, input_brute_force, step, model_path_rnn)
            wellbeing_differences_brute_force[step].append(np.mean(prediction_brute_force - real_future_step) - bias)

            prediction_max_ac = predict_n_step_rnn(index_row, input_max_ac, step, model_path_rnn)
            wellbeing_differences_max_ac[step].append(np.mean(prediction_max_ac - real_future_step) - bias)

            prediction_no_emi = predict_n_step_rnn(index_row, np.zeros(n_inputs), step, model_path_rnn)
            wellbeing_differences_no_emi[step].append(np.mean(prediction_no_emi - real_future_step) - bias)

            if step == step_x:
                ratios_trainset_online["valid ratio"].append(myutils.get_valid_ratio(X_rnn[:index + 1, :]))
                ratios_trainset_online["trainset length"].append(len(X_rnn[:index + 1, :]))
                ratios_trainset_online["optimal control"].append(np.mean(prediction_opt_ctrl - real_future_step))
                ratios_trainset_online["brute force"].append(np.mean(prediction_brute_force - real_future_step))
                ratios_trainset_online["max ac"].append(np.mean(prediction_max_ac - real_future_step))
                ratios_trainset_online["real emi"].append(np.mean(prediction_real_emi - real_future_step))
                ratios_trainset_online["no emi"].append(np.mean(prediction_no_emi - real_future_step))

    for step in n_steps:
        
        wellbeing_differences_opt_ctrl_mean = np.mean(wellbeing_differences_opt_ctrl[step])
        wellbeing_differences_brute_force_mean = np.mean(wellbeing_differences_brute_force[step])
        wellbeing_differences_max_ac_mean = np.mean(wellbeing_differences_max_ac[step])
        wellbeing_differences_real_emi_mean = np.mean(wellbeing_differences_real_emi[step])
        wellbeing_differences_no_emi_mean = np.mean(wellbeing_differences_no_emi[step])

        bias_rnn_mean = np.mean(bias_rnn[step])

        print(f'Prediction Length (step): {step}')
        print(f'opt_ctrl difference: {wellbeing_differences_opt_ctrl_mean}')
        print(f'brute_force difference: {wellbeing_differences_brute_force_mean}')
        print(f'max_ac difference: {wellbeing_differences_max_ac_mean}')
        print(f'real emi difference: {wellbeing_differences_real_emi_mean}')
        print(f'no emi difference: {wellbeing_differences_no_emi_mean}')
        print()

        if step == 1:  # Store filenames only once per dataset
            filenames.append(file)

        overall_wellbeing_change_opt_ctrl[step].append(wellbeing_differences_opt_ctrl_mean)
        overall_wellbeing_change_brute_force[step].append(wellbeing_differences_brute_force_mean)
        overall_wellbeing_change_max_ac[step].append(wellbeing_differences_max_ac_mean)
        overall_wellbeing_change_real_emi[step].append(wellbeing_differences_real_emi_mean)
        overall_wellbeing_change_no_emi[step].append(wellbeing_differences_no_emi_mean)

        overall_bias_rnn[step].append(bias_rnn_mean)
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
    data[f"bias (n={step})"] = overall_bias_rnn.get(step, [])

df_results = pd.DataFrame(data)
save_path_results = os.path.join(save_folder, filename_results)
df_results.to_csv(save_path_results, index=False)
print(f"Results saved to {save_path_results}")

savepath_results_data_quality = os.path.join(save_folder, filename_results[:-4] + filename_results_data_quality)
df_ratio_length = pd.DataFrame(ratios_trainset_online)
df_ratio_length.to_csv(savepath_results_data_quality, index=False)
print(f"Ratio and Trainsetlength results saved to {savepath_results_data_quality}")