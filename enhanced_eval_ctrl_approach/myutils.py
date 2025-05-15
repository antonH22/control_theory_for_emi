import os
import glob
import numpy as np
import pandas as pd
import torch as tc
import re

from ctrl import discrete_optimal_control as doc
from ctrl import utils

### Data utils

def csv_to_dataset(file_path, state_columns, input_columns, centered=True, remove_initial_nan=True):
    " Load a CSV file, adjust data and convert it to a datset (dictionary). "
    csv_df = pd.read_csv(file_path)
    required_columns = state_columns + input_columns
    csv_df = csv_df[required_columns]
    
    if remove_initial_nan:
        # Delete empty rows in the beginning
        first_non_na_index = csv_df.notna().all(axis=1).idxmax()
        csv_df = csv_df.iloc[first_non_na_index:].reset_index(drop=True)
    
    # Split into state and input variables (ndarrays)
    X = csv_df[state_columns].values
    Inp = csv_df[input_columns].values

    # Center state variables to [-3, 3]
    if centered:
        X -= 4
    return {'X': X, 'Inp': Inp}

def load_dataset(data_folder, subfolders, state_columns, input_columns, centered=True, remove_initial_nan=True):
    " Load all CSV files from the given subfolders and return dataset list and filenames. "
    dataset_list = []
    files = []
    for subfolder in subfolders:
        folder_path = os.path.join(data_folder, subfolder, "*.csv")
        for file in glob.glob(folder_path):
            data = csv_to_dataset(file, state_columns, input_columns, centered, remove_initial_nan)
            dataset_list.append(data)
            files.append(file)
    return dataset_list, files

def dataset_to_csv(dataset, state_columns, input_columns, output_file):
    " Convert a dataset (dictionary) back to a CSV file. "
    # Extract the state and input matrices from the dataset
    X = dataset['X']
    Inp = dataset['Inp']
    
    # Concatenate the state and input arrays horizontally
    data = np.hstack((X, Inp))

    columns = state_columns + input_columns
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_file, index=False)

def locf(X_train):
    " Imputes missing values using Last Observation Carried Forward (LOCF)."
    df_helper_locf = pd.DataFrame(X_train).copy()
    df_helper_locf.ffill(inplace=True)
    X_train_locf = df_helper_locf.to_numpy()
    return X_train_locf

def get_valid_rows(X):
    " Returns a boolean mask indicating valid rows (valid row: both the current and next row contain no NaN values.)"
    valid = ~np.isnan(X).any(axis=1)
    valid_rows = valid[:-1] & valid[1:]
    return valid_rows

def get_valid_ratio(X):
    " Computes the ratio of valid rows "
    valid_rows = get_valid_rows(X)
    valid_ratio = valid_rows.sum() / len(X)
    return valid_ratio

### Metrics utils used for LDS reliability and robustness analysis

def frobenius_norm(matrix):
    " Computes the Frobenius norm (element-wise L2 norm) of a matrix. "
    return np.sqrt(np.sum(np.abs(matrix) ** 2))

def compute_model_norms(X_train, U_train):
    " Computtes Frobenius norms of system matrix A, Kalman gain K, and L2 norm of average controllability. "
    X_train_locf = locf(X_train)
    A, B, lmbda = utils.stable_ridge_regression(X_train_locf, U_train)
    frobenius_norm_A = frobenius_norm(A)

    Q = np.eye(15)
    R = np.eye(8)
    # Compute the optimal gain matrix K
    try: 
        K = doc.kalman_gain(A, B, Q, R)
    except np.linalg.LinAlgError as e:
        print(f"Warning: Failed to solve DARE: {e}")
        return False
    
    frobenius_norm_K = frobenius_norm(K)

    ac_per_ema = doc.average_ctrb(A)
    l2norm_AC = np.linalg.norm(ac_per_ema)
    return frobenius_norm_A, frobenius_norm_K, l2norm_AC

def prediction_error(X_train, U_train, X_test, U_test):
    " Computes the MAE of one-step predictions on test data, skipping NaN rows. "
    X_train_locf = locf(X_train)
    A, B, lmbda = utils.stable_ridge_regression(X_train_locf, U_train)
    mae_per_step_list = []
    
    for i in range(len(X_test) -1):
        if np.isnan(X_test[i]).any() or np.isnan(X_test[i + 1]).any():
            continue
        x_next = doc.step(A, B, X_test[i], U_test[i])
        mae_per_step = np.mean(np.abs(x_next - X_test[i+1]))
        mae_per_step_list.append(mae_per_step)

    if mae_per_step_list:
        return np.mean(mae_per_step_list)
    else:
        return False

def compute_dominant_eigenvalue(X_train, U_train):
    " Computes the largest absolute eigenvalue of the inferred system matrix A. "
    X_train_locf = locf(X_train)
    A, B, lmbda = utils.stable_ridge_regression(X_train_locf, U_train)
    eigenvalues = np.linalg.eigvals(A)
    dominant_eigenvalue = np.max(np.abs(eigenvalues))
    return dominant_eigenvalue

### Utils for comparing rnn with lds

from bptt.plrnn import PLRNN

def prediction_error_rnn(model_path, now, step_by_step, n_steps, X, U):
    " Computes step-wise or multi-step MAE for n_steps RNN predictions from timepoint now, averaged across 10 model initializations. "
    X = tc.from_numpy(X).to(dtype=tc.float32)
    U = tc.from_numpy(U).to(dtype=tc.float32)

    # Initialize accumulator for MAE values per step across models
    mae_accumulator = [[] for _ in range(n_steps)]

    X_test = X[now+1:now+n_steps+1]  # Test data (non-smoothed)

    for model_iter in range(10):
        model_nr = str(model_iter + 1).zfill(3)
        model_path_specific = os.path.join(model_path, model_nr)
        try:
            model = PLRNN(load_model_path=model_path_specific)
        except AssertionError as e:
            print(f"Error: {e}. No model found at {model_path_specific}. Exiting function.")
            return [], [], []

        # Generate model predictions
        predictions_list = []
        if step_by_step:
            current_row = now
            for step in range(n_steps):
                if current_row >= X.shape[0]:
                    break
                if tc.isnan(X[current_row]).any():
                    nan_vector = tc.full_like(X[current_row], float('nan')).unsqueeze(0)
                    predictions_list.append(nan_vector)
                else:
                    prediction = model.generate_free_trajectory(
                        X[current_row], 1, inputs=U[current_row:current_row+1]
                    )
                    predictions_list.append(prediction)
                current_row += 1
        else:
            predictions = model.generate_free_trajectory(
                X[now], n_steps, inputs=U[now:now+n_steps]
            )
            predictions_list.append(predictions)

        predictions_tensor = tc.cat(predictions_list, dim=0)

        # Compute the MAE for each step
        model_mae = []
        for step in range(len(X_test)):
            if step >= predictions_tensor.shape[0]:
                mae = float('nan')
            else:
                mae = tc.mean(tc.abs(predictions_tensor[step] - X_test[step])).item()
            model_mae.append(mae)

        # Accumulate MAE values for each step across models
        for step_idx in range(n_steps):
            if step_idx < len(model_mae):
                mae_accumulator[step_idx].append(model_mae[step_idx])
            else:
                mae_accumulator[step_idx].append(float('nan'))

    # Compute the average MAE per step across models, ignoring NaN values
    mae_per_step_list = [np.nanmean(steps) if len(steps) > 0 else float('nan') for steps in mae_accumulator]
    return mae_per_step_list

def prediction_error_lds(now, step_by_step, n_steps, X, U):
    " Computes step-wise or multi-step MAE for n_steps LDS predictions from timepoint now. "
    mae_per_step_list = []
    # Split data
    X_train = X[:now]
    U_train = U[:now]
    X_test = X[now+1:now+n_steps+1]
    
    X_train_locf = locf(X_train)
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
    # Compute the MAE for each step
    for i in range(len(X_test)):
        mae_per_step = np.mean(np.abs(predictions_np[i] - X_test[i]))
        mae_per_step_list.append(mae_per_step)
    return mae_per_step_list

### RNN model file path extraction utils

def get_model_paths(participant_nr, folder_path):
    " Maps timesteps to model paths for a participant using filename pattern matching. "
    model_paths = {}
    for filename in os.listdir(folder_path):
        match = re.search(rf'data_\d+_\d+\.csv_participant_{participant_nr}_date_(\d+(?:\.\d+)?)', filename)
        if match:
            # The timesteps need to be subtracted by two so that they correspond to the last time point of a day
            timestep = int(float(match.group(1))) - 2
            model_paths[timestep] = os.path.join(folder_path, filename)
    return model_paths

def get_csv_file_path(participant_nr, folder_path):
    " Finds the CSV data file for a specific participant in the given directory. "
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv") and filename.split("_")[1].split(".")[0] == str(participant_nr):
            return os.path.join(folder_path, filename)
    return None

def load_data(participant_nr, data_directory, centered):
    " Loads EMA and input data for a participant, with optional centering (for LDS). "
    ema_labels = ['EMA_mood', 'EMA_disappointed', 'EMA_scared', 'EMA_worry', 'EMA_down', 'EMA_sad', 'EMA_confidence', 'EMA_stress', 'EMA_lonely', 'EMA_energetic', 'EMA_concentration', 'EMA_resilience', 'EMA_tired', 'EMA_satisfied', 'EMA_relaxed']
    input_labels = ['interactive1', 'interactive2', 'interactive3', 'interactive4', 'interactive5', 'interactive6', 'interactive7', 'interactive8']

    csv_path = get_csv_file_path(participant_nr, data_directory)
    if csv_path is None:
        raise ValueError(f"No CSV file found for participant {participant_nr} in {data_directory}")
    
    data = csv_to_dataset(csv_path, ema_labels, input_labels, centered=centered, remove_initial_nan=False)
    X, U = data['X'], data['Inp']
                                                                                                                                
    return X, U

def extract_participant_ids(folder_path):
    " Extracts unique participant IDs from filenames in a directory. "
    participant_ids = set()
    for filename in os.listdir(folder_path):
        match = re.search(r'participant_(\d+)', filename)
        if match:
            participant_ids.add(int(match.group(1)))
    return sorted(participant_ids)

def extract_participant_nr_and_model_path(file, rnn_model_path_MRT1, rnn_model_path_MRT2, rnn_model_path_MRT3):
    " Extracts participant number and corresponding model path from filename"
    match = re.search(r"MRT(\d)", file)
    mrt_nr = int(match.group(1))
    model_paths = {1: rnn_model_path_MRT1, 2: rnn_model_path_MRT2, 3: rnn_model_path_MRT3}
    model_path = model_paths.get(mrt_nr)
    # Get participant number
    match = re.search(r'_(\d+)\.csv$', file)
    participant_nr = int(match.group(1))
    return participant_nr, model_path

def get_latest_model_path(participant_nr, model_path, index):
    " Returns the path of the latest available RNN model for a participant up to index. "
    model_paths_dict = get_model_paths(participant_nr, model_path)
    if not model_paths_dict:
        return False
    # Select latest model up to index
    valid_keys = [k for k in model_paths_dict.keys() if k <= index]
    if not valid_keys:
        # Then there is no trained model
        return False
    key_online = max(valid_keys)
    model_path_scenario = model_paths_dict[key_online]
    return model_path_scenario

### Utils for inverted pendulum

def simulate_system_step_nonlinear(x, u, noise_std=0.0, delta_t=0.01):
    " Nonlinear dynamics simulation function for the inverted pendulum "
    m = 1  # mass of the pendulum
    M = 1  # mass of the cart
    L = 2  # length of the pendulum arm
    g = -10  # gravitational acceleration
    d = 1  # damping factor

    Sx = np.sin(x[2])  # sin(theta)
    Cx = np.cos(x[2])  # cos(theta)
    D = m * L**2 * (M + m * (1 - Cx**2))

    dx = np.zeros(4)

    u = u[0] # Control input (one element array)

    # State equations
    dx[0] = x[1]  # x_dot
    dx[1] = (1 / D) * (-m**2 * L**2 * g * Cx * Sx + m * L**2 * (m * L * x[3]**2 * Sx - d * x[1])) + m * L**2 * (1 / D) * u
    dx[2] = x[3]  # theta_dot
    dx[3] = (1 / D) * ((M + m) * m * g * L * Sx - m * L * Cx * (m * L * x[3]**2 * Sx - d * x[1])) - m * L * Cx * (1 / D) * u

    # Add noise to specific nodes
    noise = np.random.normal(0, noise_std)
    noise_angle = np.array([0, 0, noise, 0])

    x_next = x + dx * delta_t + noise_angle  # Assuming small time step (Euler method)
    
    return x_next

def simulate_pendulum(x_0, x_ref, A, B, noise_std, num_steps):
    " Simulates pendulum under LQR control. "
    x_t = x_0
    Q = np.diag([1, 1, 1, 1])  # State cost matrix
    R = np.array([[1]]) # Control effort cost matrix (large values reduce control effort)
    
    # Compute the optimal gain matrix K
    K = doc.kalman_gain(A, B, Q, R)
    states = [x_0]
    inputs = []
    for i in range(num_steps):
        # Compute the optimal control input using the Kalman gain
        u = -K @ (x_t - x_ref)
        # Update the state using the nonlinear or linear dynamics function
        x_t = simulate_system_step_nonlinear(x_t, u, noise_std)  # Get the next state
        states.append(x_t)
        inputs.append(u)
    return states, inputs
