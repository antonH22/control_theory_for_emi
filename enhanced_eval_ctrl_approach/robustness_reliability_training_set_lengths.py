import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ctrl import discrete_optimal_control as doc
from ctrl import utils
import numpy as np
import pandas as pd
import json

### Analyzing robustness and reliability of the LDS model across training set lengths (5.1.4, Figure 4)

emas = ['EMA_mood', 'EMA_disappointed', 'EMA_scared', 'EMA_worry', 'EMA_down', 'EMA_sad', 'EMA_confidence', 'EMA_stress', 'EMA_lonely', 'EMA_energetic', 'EMA_concentration', 'EMA_resilience', 'EMA_tired', 'EMA_satisfied', 'EMA_relaxed']
emis = ['interactive1', 'interactive2', 'interactive3', 'interactive4', 'interactive5', 'interactive6', 'interactive7', 'interactive8']

data_folder = "data"
subfolders = ["MRT1/processed_csv_no_con","MRT2/processed_csv_no_con","MRT3/processed_csv_no_con"]

# Set parameters
split = 0.7 # Not on basis of the valid ratio, but total length (different from the ratio analysis scripts)
trainlens = [20, 40, 60, 80, 100, 120, 140, 160]

def num_valid_rows(data):
    valid = ~np.isnan(data).any(axis=1)
    valid_rows = valid[:-1] & valid[1:]
    total = valid_rows.sum()
    return total

def get_training_set(trainlen, X_train, U_train):
    number_to_remove = len(X_train) - trainlen
    if number_to_remove < 0:
        return False
    X_train_new = X_train[number_to_remove:]
    U_train_new = U_train[number_to_remove:]
    if num_valid_rows(X_train_new) == 0:
        return False
    return X_train_new, U_train_new

def compute_eigenvalues(X_train_ratio, U_train_ratio):
    A, B, lmbda = utils.stable_ridge_regression(X_train_ratio, U_train_ratio)
    eigenvalues = np.linalg.eigvals(A)
    return eigenvalues
    
def frobenius_norm(matrix):
    return np.sqrt(np.sum(np.abs(matrix) ** 2))

def locf(X_train): 
    df_helper_locf = pd.DataFrame(X_train).copy()
    df_helper_locf.ffill(inplace=True)
    X_train_locf = df_helper_locf.to_numpy()
    return X_train_locf

def norm_per_trainlen(X_train_trainlen, U_train_trainlen):
    X_train_locf = locf(X_train_trainlen)
    A, B, lmbda = utils.stable_ridge_regression(X_train_locf, U_train_trainlen)
    frobenius_norm_A = frobenius_norm(A)

    Q = np.eye(len(emas))
    R = np.eye(len(emis))
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

def prediction_error(X_train_trainlen, U_train_trainlen, X_test, U_test):
    X_train_locf = locf(X_train_trainlen)
    A, B, lmbda = utils.stable_ridge_regression(X_train_locf, U_train_trainlen)
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

def compute_dominant_eigenvalue(X_train_trainlen, U_train_trainlen):
    X_train_locf = locf(X_train_trainlen)
    A, B, lmbda = utils.stable_ridge_regression(X_train_locf, U_train_trainlen)
    eigenvalues = np.linalg.eigvals(A)
    dominant_eigenvalue = np.max(np.abs(eigenvalues))
    return dominant_eigenvalue

def get_num_valid_rows(data):
    # Valid rows are rows without nan value where also the next row has no nan value
    valid = ~np.isnan(data).any(axis=1)
    valid_rows = valid[:-1] & valid[1:]
    total = valid_rows.sum()
    return total

# Dictionaries to store trainlens and corresponding frobenius norms
results_A = {trainlen: [] for trainlen in trainlens}
results_K = {trainlen: [] for trainlen in trainlens}
results_AC = {trainlen: [] for trainlen in trainlens}
results_corr_A = {trainlen: [] for trainlen in trainlens}
results_mae = {trainlen: [] for trainlen in trainlens}
results_eigenvalues = {trainlen: [] for trainlen in trainlens}

dataset_list, files = utils.load_dataset(data_folder, subfolders, emas, emis, centered=True)

skip_files = {}
num_analysed_files = 0

with open('exclude_participant_list.json', 'r') as f:
    exclude_participant_list = json.load(f)

num_analysed_files = 0
for dataset, file in zip(dataset_list, files):
    if file in exclude_participant_list:
        # Participants for whom there is no trained RNN model are excluded from all analyses
        continue
    X, U = dataset['X'], dataset['Inp']

    num_analysed_files += 1
    print(f'Loop {num_analysed_files}/143')

    # Determine the split index for the training and testing data
    split_index = int(np.floor(len(X) * 0.7))

    # Split data
    X_train, X_test = X[:split_index], X[split_index:]
    U_train, U_test = U[:split_index], U[split_index:]

    if len(X_train) < 160:
        continue

    matrices_A = {}

    skip_iteration = False
    #print(f'Valid ratio train {valid_train_ratio}')
    for trainlen in trainlens:
        if len(X_train) < trainlen:
            continue
        resulting_training_set = get_training_set(trainlen, X_train, U_train)
        if resulting_training_set == False:
            skip_iteration = True
            break
        X_train_trainlen, U_train_trainlen = resulting_training_set
        resulting_norms = norm_per_trainlen(X_train_trainlen, U_train_trainlen) 
        frobenius_norm_A, frobenius_norm_K, l2_norm_AC = resulting_norms
        results_A[trainlen].append(frobenius_norm_A)
        results_K[trainlen].append(frobenius_norm_K)
        results_AC[trainlen].append(l2_norm_AC)

        mae = prediction_error(X_train_trainlen, U_train_trainlen, X_test, U_test)
        if mae:
            # Mae can only be computed if there is valid test data
            results_mae[trainlen].append(mae)

        dominant_eigenvalue = compute_dominant_eigenvalue(X_train_trainlen, U_train_trainlen)
        results_eigenvalues[trainlen].append(dominant_eigenvalue)

        A, B, lmbda = utils.stable_ridge_regression(X_train_trainlen, U_train_trainlen)
        matrices_A[trainlen] = A
        #print(f'Trainlen: {trainlen} real trainlen {len(X_train_trainlen)}')

    if skip_iteration:
        continue
    # Compute correlations of each A matrix with A matrix inferred on trainlen 160
    A_base = matrices_A[160]
    for trainlen in trainlens:
        if trainlen not in matrices_A:
            continue
        A_comp = matrices_A[trainlen]
        # Compute Pearson correlation between two flattened matrices
        corr_A = np.corrcoef(A_comp.flatten(), A_base.flatten())[0, 1]
        if not np.isnan(corr_A):
            results_corr_A[trainlen].append(corr_A)
    
# Compute mean, standard deviation, standard errors for each trainlen
def compute_me_sd_se(results_dict):
    mean_errors = [np.mean(errors) for errors in results_dict.values()]
    sd_errors = [np.std(errors) for errors in results_dict.values()]
    # Compute std error
    num_elements = [len(errors) for errors in results_dict.values()]
    se_errors = [0] * len(trainlens)  # Initialize se_errors as a list of zeros
    for i,_ in enumerate(trainlens):
        #print(f'Number of samples for valid ratio {trainlens[i]}: {num_elements[i]}')
        se_errors[i] = sd_errors[i] / np.sqrt(num_elements[i])
    return mean_errors, sd_errors, se_errors

# Define metrics and their labels
metrics = [
    (results_A, "frobenius_A", r"$||A||_F$"),
    (results_K, "frobenius_K", r"$||K||_F$"),
    (results_AC, "frobenius_AC", r"$L_2(\mathrm{AC})$"),
    (results_corr_A, "corr_A", r"$r(A, A_{0.8})$"),
    (results_mae, "mae", "MAE"),
    (results_eigenvalues, "eigenvalue_A", r"$\lambda_1(A)$")
]

folder = "results_replicated"
combined_results = []

for result, metric_name, label in metrics:
    mean, sd, se = compute_me_sd_se(result)
    for tl, m, s, e in zip(trainlens, mean, sd, se):
        combined_results.append({
            "trainlen": tl,
            "metric": metric_name,
            "mean": m,
            "sd": s,
            "se": e,
            "ylabel": label
        })

# Save all results to a single CSV
combined_df = pd.DataFrame(combined_results)
combined_df.to_csv(os.path.join(folder, "results_training_set_lengths_metrics.csv"), index=False)