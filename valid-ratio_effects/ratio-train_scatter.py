import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ctrl import discrete_optimal_control as doc
from ctrl import utils
import numpy as np
import glob
import pandas as pd

emas = ['EMA_mood', 'EMA_disappointed', 'EMA_scared', 'EMA_worry', 'EMA_down', 'EMA_sad', 'EMA_confidence', 'EMA_stress', 'EMA_lonely', 'EMA_energetic', 'EMA_concentration', 'EMA_resilience', 'EMA_tired', 'EMA_satisfied', 'EMA_relaxed']
emis = ['interactive1', 'interactive2', 'interactive3', 'interactive4', 'interactive5', 'interactive6', 'interactive7', 'interactive8']

data_folder = "data"
subfolders = ["MRT1/processed_csv_no_con","MRT2/processed_csv_no_con","MRT3/processed_csv_no_con"]

# Set parameters
split = 1.0
num_rows_threshold = 50 # One file is excluded
num_resubsampling = 1 # Number of times the training set creation is repeated to reduce standard deviation caused by individual random removal process
ratio_reduction = 0.1

# Set the filename of the results
filename = f'completeratio-everything{num_resubsampling}_locf_reduction.csv'

def remove_one_valid_row(data, input):
    # Find indices of one non nan row and delete one randomly
    valid_rows = np.where(~np.isnan(data).any(axis=1))[0] 
    row_to_remove = np.random.choice(valid_rows)
    data = np.delete(data, row_to_remove, axis=0)
    input = np.delete(input, row_to_remove, axis = 0)
    return data, input

def get_valid_ratio(data):
    # Valid rows are rows without nan value where also the next row has no nan value
    valid = ~np.isnan(data).any(axis=1)
    valid_rows = valid[:-1] & valid[1:]
    total = valid_rows.sum()
    return total/len(data)

def get_training_set(target_ratio, X_train, U_train):
    # Returns the training set with target ratio
    current_ratio = get_valid_ratio(X_train)
    X_current, U_current = X_train, U_train

    # Initialize last valid state as the starting state
    X_last, U_last, ratio_last = X_train, U_train, current_ratio

    while current_ratio > target_ratio:
        X_current, U_current = remove_one_valid_row(X_current, U_current)
        current_ratio = get_valid_ratio(X_current)
        # Update last valid state if the new ratio is closer to target_ratio
        if abs(current_ratio - target_ratio) < abs(ratio_last - target_ratio):
            X_last, U_last, ratio_last = X_current, U_current, current_ratio

    return X_last, U_last, ratio_last

def locf(X_train): 
    df_helper_locf = pd.DataFrame(X_train).copy()
    df_helper_locf.ffill(inplace=True)
    X_train_locf = df_helper_locf.to_numpy()
    return X_train_locf

def compute_eigenvalues(X_train_ratio, U_train_ratio):
    X_train_ratio = locf(X_train_ratio)
    A, B, lmbda = utils.stable_ridge_regression(X_train_ratio, U_train_ratio)
    eigenvalues = np.linalg.eigvals(A)
    return eigenvalues

def frobenius_norm(matrix):
    return np.sqrt(np.sum(np.abs(matrix) ** 2))

def frobenius_norms(X_train_ratio, U_train_ratio):
    X_train_ratio = locf(X_train_ratio)
    A, B, lmbda = utils.stable_ridge_regression(X_train_ratio, U_train_ratio)
    frobenius_norm_A = frobenius_norm(A)

    Q = np.eye(len(emas))
    R = np.eye(len(emis))
    # Compute the optimal gain matrix K
    K = doc.kalman_gain(A, B, Q, R)
    frobenius_norm_K = frobenius_norm(K)

    ac_per_ema = doc.average_ctrb(A)
    l2norm_AC = np.linalg.norm(ac_per_ema)
    return frobenius_norm_A, frobenius_norm_K, l2norm_AC

# Lists to store ratios and corresponding frobenius norms
results_ratios = []
results_eigenvalues_A = []
results_frobenius_A = []
results_frobenius_K = []
results_l2_AC = []

dataset_list = []
files = []

dataset_list, files = utils.load_dataset(data_folder, subfolders, emas, emis, centered=True)

skip_files = {}
num_analysed_files = 0
for idx, dataset in enumerate(dataset_list):
    if files[idx] in skip_files:
        continue
    X, U = dataset['X'], dataset['Inp']

    if len(X) < num_rows_threshold:
        continue
    
    # Determine the split index for the training and testing data
    valid = ~np.isnan(X).any(axis=1)
    valid_rows = valid[:-1] & valid[1:] # valid rows are those where the predictor and target are both valid (no NaN values)
    total = valid_rows.sum() # total number of valid rows
    split_index = np.searchsorted(np.cumsum(valid_rows), total * split) # searches for the index in (np.cumsum(pairs)) where the cumulative sum first exceeds 70% of the valid rows

    num_analysed_files += 1
    print(f'Loop {num_analysed_files}/176')
    #print('-' * 40)
    #print(f'{files[idx]}:')
    #print(f'Split index: {split_index}')
    
    # Split data
    X_train, X_test = X[:split_index], X[split_index:]
    U_train, U_test = U[:split_index], U[split_index:]

    initial_ratio = get_valid_ratio(X_train)

    X_train_locf = locf(X_train)
    eigenvalues = compute_eigenvalues(X_train_locf, U_train)
    dominant_eigenvalue = np.max(np.abs(eigenvalues))
    results_ratios.append(initial_ratio)
    results_eigenvalues_A.append(dominant_eigenvalue)

    frobenius_A, frobenius_K, l2_AC = frobenius_norms(X_train, U_train)
    results_frobenius_A.append(frobenius_A)
    results_frobenius_K.append(frobenius_K)
    results_l2_AC.append(l2_AC)
    
    #print(f'Valid ratio train {valid_train_ratio}')
    iterations = 1
    target_ratio = initial_ratio - ratio_reduction
    print()
    print(files[idx])
    while target_ratio > 0.1:
        skip = False
        # Do multiple ratioed dataset creations -> multiple different removel processes -> decrease standard deviation caused by individual random removal process
        target_ratio = initial_ratio - ratio_reduction * iterations
        iterations += 1
        #ratios_resamp = []
        #eigenvalues_resamp = []

        # Does a resampling loop make sense here?
        X_train_ratio, U_train_ratio, current_ratio = get_training_set(target_ratio, X_train, U_train)
        if current_ratio == 0.0:
            skip = True
            continue
            
        eigenvalues = compute_eigenvalues(X_train_ratio, U_train_ratio)
        dominant_eigenvalue = np.max(np.abs(eigenvalues))
        results_ratios.append(initial_ratio)
        results_eigenvalues_A.append(dominant_eigenvalue)

        frobenius_A, frobenius_K, l2_AC = frobenius_norms(X_train_ratio,U_train_ratio)
        results_frobenius_A.append(frobenius_A)
        results_frobenius_K.append(frobenius_K)
        results_l2_AC.append(l2_AC)
        print(f'Ratio: {current_ratio}; dominant eigenvalue: {dominant_eigenvalue}')
    
        
        if skip:
            print("While loop broke: new participant iteration")
            break
        # When resampling the training sets, take the mean over all samples per ratio reduction
        #mean_ratio_resamp = sum(ratios_resamp) / len(ratios_resamp)
        #mean_eigenvalue_resamp = sum(eigenvalues_resamp) / len(eigenvalues_resamp)
        #results_ratios.append(mean_ratio_resamp)
        #results_eigenvalues.append(mean_eigenvalue_resamp)"
        
# Convert results to a DataFrame to save them to a csv file
df_final = pd.DataFrame({
    "ratio": results_ratios,
    "eigenvalues": results_eigenvalues_A,
    "frobenius_A": results_frobenius_A,
    "frobenius_K": results_frobenius_K,
    "l2_AC": results_l2_AC
})

# Save to CSV
folder = "results_ratio"
filepath = os.path.join(folder, filename)
df_final.to_csv(filepath, index=False)
print(f'Final results saved to {filename}')

print(f'Datasets analysed {num_analysed_files}')
print(f'Number of datapoints: {len(results_ratios)}')