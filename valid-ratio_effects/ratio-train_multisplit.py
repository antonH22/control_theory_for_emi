import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ctrl import discrete_optimal_control as doc
from ctrl import utils
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import re


emas = ['EMA_mood', 'EMA_disappointed', 'EMA_scared', 'EMA_worry', 'EMA_down', 'EMA_sad', 'EMA_confidence', 'EMA_stress', 'EMA_lonely', 'EMA_energetic', 'EMA_concentration', 'EMA_resilience', 'EMA_tired', 'EMA_satisfied', 'EMA_relaxed']
emis = ['interactive1', 'interactive2', 'interactive3', 'interactive4', 'interactive5', 'interactive6', 'interactive7', 'interactive8']

data_folder = "data"
subfolders = ["MRT1/processed_csv_no_con","MRT2/processed_csv_no_con","MRT3/processed_csv_no_con"]

# Set parameters
splits = [0.7, 0.75, 0.8, 0.85]
num_rows_threshold = 150
ratio_threshold = 0.8 # For each split the valid ratio needs to be greater than 80%
ratios = [0.8,0.7,0.6,0.5,0.4,0.3,0.2]
num_resubsampling = 30 # Number of times the training set creation is repeated to reduce standard deviation caused by individual random removal process

analyze_A_matrix = False

fileA = 'data\\MRT1/processed_csv_no_con\\11228_34.csv'
fileA2 = 'data\\MRT1/processed_csv_no_con\\11228_52.csv'

# Exclude files for which you cant get trainingsets for all ratios
#skip_files = ['data\MRT1/processed_csv_no_con\\11228_61.csv', 'data\MRT3/processed_csv_no_con\\12600_270.csv']
#files_ratio80 = ['data\\MRT1/processed_csv_no_con\\11228_15.csv', 'data\\MRT1/processed_csv_no_con\\11228_19.csv', 'data\\MRT1/processed_csv_no_con\\11228_28.csv', 'data\\MRT1/processed_csv_no_con\\11228_34.csv', 'data\\MRT1/processed_csv_no_con\\11228_35.csv', 'data\\MRT1/processed_csv_no_con\\11228_52.csv', 'data\\MRT2/processed_csv_no_con\\12600_30.csv', 'data\\MRT2/processed_csv_no_con\\12600_32.csv', 'data\\MRT2/processed_csv_no_con\\12600_64.csv', 'data\\MRT2/processed_csv_no_con\\12600_65.csv', 'data\\MRT3/processed_csv_no_con\\12600_218.csv', 'data\\MRT3/processed_csv_no_con\\12600_221.csv', 'data\\MRT3/processed_csv_no_con\\12600_228.csv', 'data\\MRT3/processed_csv_no_con\\12600_239.csv', 'data\\MRT3/processed_csv_no_con\\12600_241.csv', 'data\\MRT3/processed_csv_no_con\\12600_261.csv']

files_ratio80_split85 = ['data\\MRT1/processed_csv_no_con\\11228_28.csv', 'data\\MRT1/processed_csv_no_con\\11228_34.csv', 'data\\MRT1/processed_csv_no_con\\11228_35.csv', 'data\\MRT1/processed_csv_no_con\\11228_52.csv', 'data\\MRT2/processed_csv_no_con\\12600_25.csv', 'data\\MRT2/processed_csv_no_con\\12600_30.csv', 'data\\MRT2/processed_csv_no_con\\12600_32.csv', 'data\\MRT3/processed_csv_no_con\\12600_221.csv', 'data\\MRT3/processed_csv_no_con\\12600_239.csv', 'data\\MRT3/processed_csv_no_con\\12600_241.csv']
# Set the filename of the results
#filename = f'matrix_a_MRT1-35_{num_resubsampling}ratio80_splits{len(splits)}.csv'

filename = f'results_{num_resubsampling}ratio80_splits{len(splits)}_new.csv'
save_path = os.path.join("results_ratio", filename)
save_path_flatA = os.path.join("results_ratio", filename[:-4]+"_flatA.csv")
save_matrices_fileA = os.path.join("results_ratio", filename[:-4]+"_matrixA_MRT1-34.csv")
save_matrices_fileA2 = os.path.join("results_ratio", filename[:-4]+"_matrixA_MRT1-52.csv")


def remove_one_valid_row(data, input):
    # Create copies of the data to avoid side effects
    data_copy = data.copy()
    input_copy = input.copy()
    
    # Find indices of rows that do not contain NaN values and delete one randomly
    valid_rows = np.where(~np.isnan(data_copy).any(axis=1))[0]
    row_to_remove = np.random.choice(valid_rows)
    data_copy = np.delete(data_copy, row_to_remove, axis=0)
    input_copy = np.delete(input_copy, row_to_remove, axis=0)
    return data_copy, input_copy

def get_valid_ratio(data):
    # Valid rows are rows without nan value where also the next row has no nan value
    valid = ~np.isnan(data).any(axis=1)
    valid_rows = valid[:-1] & valid[1:]
    total = valid_rows.sum()
    return total/len(data)

def get_training_set(valid_train_ratio, target_ratio, X_train, U_train):
    # Initialize current state with the original training sets
    X_current, U_current = X_train, U_train
    current_ratio = get_valid_ratio(X_train)

    # Initialize last valid state as the starting state
    X_last, U_last, ratio_last = X_train, U_train, valid_train_ratio

    while current_ratio > target_ratio:
        X_current, U_current = remove_one_valid_row(X_current, U_current)
        current_ratio = get_valid_ratio(X_current)
        # Update last valid state if the new ratio is closer to target_ratio
        if abs(current_ratio - target_ratio) < abs(ratio_last - target_ratio):
            X_last, U_last, ratio_last = X_current, U_current, current_ratio

    # Check if the best ratio difference is within the acceptable threshold
    if abs(ratio_last - target_ratio) > 0.05:
        print("Threshold is reached")
        return False
    else:
        return X_last, U_last
    
def locf(X_train): 
    df_helper_locf = pd.DataFrame(X_train).copy()
    df_helper_locf.ffill(inplace=True)
    X_train_locf = df_helper_locf.to_numpy()
    return X_train_locf

def prediction_error(X_train_ratio, U_train_ratio, X_test, U_test):
    X_train_ratio_locf = locf(X_train_ratio)
    A, B, lmbda = utils.stable_ridge_regression(X_train_ratio_locf, U_train_ratio)
    mae_per_step_list = []
    
    for i in range(len(X_test) -1):
        if np.isnan(X_test[i]).any() or np.isnan(X_test[i + 1]).any():
            continue
        x_next = doc.step(A, B, X_test[i], U_test[i])
        mae_per_step = np.mean(np.abs(x_next - X_test[i+1]))
        mae_per_step_list.append(mae_per_step)
    return sum(mae_per_step_list) / len(mae_per_step_list)

def mac(X_train_ratio):
    # Compute the mean absolute changes, so the absolute difference between subsequent rows (only rows that are not nan)
    df = pd.DataFrame(X_train_ratio).copy()
    df_diff = df.diff().abs()
    mac_series = df_diff.mean(axis=1)
    # Drop NaN values using dropna()
    mac_series = mac_series.dropna()
    mac_mean = mac_series.mean()
    return mac_mean

def compute_dominant_eigenvalue(X_train_ratio, U_train_ratio):
    X_train_ratio = locf(X_train_ratio)
    A, B, lmbda = utils.stable_ridge_regression(X_train_ratio, U_train_ratio)
    eigenvalues = np.linalg.eigvals(A)
    dominant_eigenvalue = np.max(np.abs(eigenvalues))
    return dominant_eigenvalue

def get_A_matrix(X_train_ratio, U_train_ratio):
    X_train_ratio = locf(X_train_ratio)
    A, B, lmbda = utils.stable_ridge_regression(X_train_ratio, U_train_ratio)
    return A

def frobenius_norm(matrix):
    return np.sqrt(np.sum(np.abs(matrix) ** 2))

def frobenius_norm_difference(A):
    n = A.shape[0]  # Get matrix size
    I = np.eye(n)   # Identity matrix
    return np.sqrt(np.sum(np.abs(A - I) ** 2))

def norm_per_ratio(X_train_ratio, U_train_ratio):
    X_train_ratio = locf(X_train_ratio)
    A, B, lmbda = utils.stable_ridge_regression(X_train_ratio, U_train_ratio)
    frobenius_norm_A = frobenius_norm_difference(A)

    Q = np.eye(len(emas))
    R = np.eye(len(emis))
    # Compute the optimal gain matrix K
    K = doc.kalman_gain(A, B, Q, R)
    frobenius_norm_K = frobenius_norm(K)

    ac_per_ema = doc.average_ctrb(A)
    l2norm_AC = np.linalg.norm(ac_per_ema)
    return frobenius_norm_A, frobenius_norm_K, l2norm_AC

###################################################################################################################

dataset_list, files = utils.load_dataset(data_folder, subfolders, emas, emis, centered=True)

# Lists of dictionaries to store ratios and corresponding errors for each participant
filenames = []
results_mae_dicts = []
results_mac_dicts = []
results_eigenvalue_dicts = []
results_frobeniusA_dicts = []
results_frobeniusK_dicts = []
results_frobeniusAC_dicts = []

results_matrixA_dicts = []

filenames = []
for idx, dataset in enumerate(dataset_list):
    if analyze_A_matrix:
        if files[idx] != 'data\\MRT1/processed_csv_no_con\\11228_35.csv':
            continue

    if files[idx] not in files_ratio80_split85:
        continue
    X, U = dataset['X'], dataset['Inp']

    if len(X) < num_rows_threshold:
        continue

    results_mae = {ratio: [] for ratio in ratios}
    results_mac = {ratio: [] for ratio in ratios}
    results_eigenvalue = {ratio: [] for ratio in ratios}
    results_frobeniusA = {ratio: [] for ratio in ratios}
    results_frobeniusK = {ratio: [] for ratio in ratios}
    results_frobeniusAC = {ratio: [] for ratio in ratios}
    results_matrixA = {ratio: [] for ratio in ratios}

    matrices = {ratio: [] for ratio in ratios}

    for split in splits:
        # Determine the split index for the training and testing data
        valid = ~np.isnan(X).any(axis=1)
        valid_rows = valid[:-1] & valid[1:] # valid rows are those where the predictor and target are both valid (no NaN values)
        total = valid_rows.sum() # total number of valid rows
        split_index = np.searchsorted(np.cumsum(valid_rows), total * split) # searches for the index in (np.cumsum(pairs)) where the cumulative sum first exceeds 70% of the valid rows

        #print('-' * 40)
        #print(f'{files[idx]}:')
        #print(f'Split index: {split_index}')
        
        # Split data
        X_train, X_test = X[:split_index], X[split_index:]
        U_train, U_test = U[:split_index], U[split_index:]

        valid_train_ratio = get_valid_ratio(X_train)
        if valid_train_ratio < ratio_threshold:
            print("ratio is below threshold")
            continue
        
        #print(f'Valid ratio train {valid_train_ratio}')
        for ratio in ratios:
            if valid_train_ratio < ratio:
                continue
            # Do multiple ratioed dataset creations -> multiple different removel processes -> decrease standard deviation caused by individual random removal process

            mae_per_sample = []
            mac_per_sample = []
            eigenvalue_per_sample = []
            frobeniusA_per_sample = []
            frobeniusK_per_sample = []
            frobeniusAC_per_sample = []

            matrixA_per_sample = []
            for i in range(num_resubsampling):
                resulting_training_set = get_training_set(valid_train_ratio, ratio, X_train, U_train)
                if resulting_training_set == False:
                    print(files[idx])
                    print("resulting_training_set == False")
                    break
                X_train_ratio, U_train_ratio = resulting_training_set
                mean_mae_ratio = prediction_error(X_train_ratio, U_train_ratio, X_test, U_test)
                mae_per_sample.append(mean_mae_ratio)
                mean_mac_ratio = mac(X_train_ratio)
                mac_per_sample.append(mean_mac_ratio)
                eigenvalue_ratio = compute_dominant_eigenvalue(X_train_ratio, U_train_ratio)
                eigenvalue_per_sample.append(eigenvalue_ratio)
                frobenius_norm_A, frobenius_norm_K, l2_norm_AC = norm_per_ratio(X_train_ratio, U_train_ratio)
                frobeniusA_per_sample.append(frobenius_norm_A)
                frobeniusK_per_sample.append(frobenius_norm_K)
                frobeniusAC_per_sample.append(l2_norm_AC)

                a_matrix = get_A_matrix(X_train_ratio, U_train_ratio)
                matrixA_per_sample.append(a_matrix)

            results_mae[ratio].append(sum(mae_per_sample) / len(mae_per_sample))
            results_mac[ratio].append(sum(mac_per_sample) / len(mac_per_sample))
            results_eigenvalue[ratio].append(sum(eigenvalue_per_sample) / len(eigenvalue_per_sample))
            results_frobeniusA[ratio].append(sum(frobeniusA_per_sample) / len(frobeniusA_per_sample))
            results_frobeniusK[ratio].append(sum(frobeniusK_per_sample) / len(frobeniusK_per_sample))
            results_frobeniusAC[ratio].append(sum(frobeniusAC_per_sample) / len(frobeniusAC_per_sample))

            results_matrixA[ratio].append(np.mean(matrixA_per_sample, axis=0))
            #print(f'Valid ratio: {ratio} real valid ratio {get_valid_ratio(X_train_ratio)}; MAE: {sum(mae_per_sample) / len(mae_per_sample)} MAC: {sum(mac_per_sample) / len(mac_per_sample)}')
    filenames.append(files[idx])
    print(f'Loop {len(filenames)}/8')
    results_mae_dicts.append(results_mae)
    results_mac_dicts.append(results_mac)
    results_eigenvalue_dicts.append(results_eigenvalue)
    results_frobeniusA_dicts.append(results_frobeniusA)
    results_frobeniusK_dicts.append(results_frobeniusK)
    results_frobeniusAC_dicts.append(results_frobeniusAC)

    results_matrixA_dicts.append(results_matrixA)
    #if len(filenames) > 1:
     #   break

# Set up colors based on participants (files)
colors = plt.cm.get_cmap('tab10', len(filenames))  # Use a colormap for different participants

csv_data = []  # List to store data for CSV file

extracted_filenames = []
for file in filenames:
    # Extract the required part (e.g. MRT1-11228_28)
    # Extract the MRT number from the file path
    mrt_number = file.split('\\')[1].split('/')[0]
    # Extract the file name without extension
    file_name = os.path.basename(file).split('.')[0] 
    extracted_part = f"{mrt_number}-{file_name}"
    extracted_filenames.append(extracted_part)

# Compute correlation per participant and ratio
results_mean_A_mean_dicts = []
results_var_A_mean_dicts = []
results_matrix_A_mean_dfs = []
for matrix_dict in results_matrixA_dicts:
    df_matrix_A_mean = pd.DataFrame()
    mean_dict = {}
    variance_dict = {}
    for ratio in ratios:
        mean_matrix = np.mean(matrix_dict[ratio], axis=0)
        mean_dict[ratio] = np.mean(mean_matrix)
        variance_dict[ratio] = np.var(mean_matrix)
        df_matrix_A_mean[ratio] = mean_matrix.flatten()
    results_mean_A_mean_dicts.append(variance_dict)   
    results_var_A_mean_dicts.append(variance_dict)
    results_matrix_A_mean_dfs.append(df_matrix_A_mean)


# Iterate through each file (participant) and compute the means per ratio
for i, (mae_part, mac_part, eigenvalue_part, frobA_part, frobK_part, frobAC_part, meanA_part, varA_part, df_flatA_part) in enumerate(zip(results_mae_dicts, results_mac_dicts, results_eigenvalue_dicts, results_frobeniusA_dicts, results_frobeniusK_dicts, results_frobeniusAC_dicts, results_mean_A_mean_dicts, results_var_A_mean_dicts, results_matrix_A_mean_dfs)):
    correlations = df_flatA_part.corrwith(df_flatA_part[0.8])
    for ratio in ratios:
        # Store data for CSV (Ratio, Mean MAE, Mean MAC, Eigenvalue, File)
        csv_data.append([
            ratio, 
            np.mean(mae_part[ratio]),
            np.mean(mac_part[ratio]),
            np.mean(eigenvalue_part[ratio]),
            np.mean(frobA_part[ratio]),
            np.mean(frobK_part[ratio]),
            np.mean(frobAC_part[ratio]),
            meanA_part[ratio],
            varA_part[ratio],
            correlations[ratio],
            extracted_filenames[i]   
        ])
        #print(f'ratio {ratio}: mean: {np.mean(mae_participant[ratio])}')

# Save results to CSV
df_results = pd.DataFrame(csv_data, columns=["Ratio", "Mean MAE","Mean MAC", "Mean Eigenvalue","Frobenius A", "Frobenius K", "Frobenius AC","Mean A","Variance A", "Correlation A", "File"])
df_results.to_csv(save_path, index=False)  # Save to CSV file
print(f'Saved all results to {save_path}')

"""
# Initialize a dictionary to store lists of matrices for each ratio
sum_matrices = {ratio: [] for ratio in results_matrixA_dicts[0].keys()}

# Collect matrices for each ratio
for matrices in results_matrixA_dicts:
    for ratio, A in matrices.items():
        #compute the mean matrix for each ratio
        sum_matrices[ratio].append(np.array(A))

mean_matrices = {ratio: np.mean(np.array(matrices_list), axis=0) for ratio, matrices_list in sum_matrices.items()}

matrix_per_ratio = {}
for ratio, matrix_list in mean_matrices.items():
    matrix_per_ratio[ratio] = (np.mean(matrix_list,axis=0))
    print(np.mean(matrix_list,axis=0).shape)

flattened_matrices = {ratio: A.flatten() for ratio, A in matrix_per_ratio.items()}
df_flattenedA = pd.DataFrame(flattened_matrices)

# Save results to CSV
df_flattenedA.to_csv(save_path_flatA, index=False)
print(f'Saved flattenedA results to {save_path_flatA}')
"""

matrixA_sample = {}
matrixA2_sample = {}
for i,file in enumerate(filenames):
    if file == fileA:
        matrixA_sample = results_matrixA_dicts[i]
    if file == fileA2:
        matrixA2_sample = results_matrixA_dicts[i]

matrices_fileA = {}
# Compute mean matrix for each ratio
for ratio, matrices_list in matrixA_sample.items():
    matrices_array = np.array(matrices_list)  # Convert list to numpy array (shape: n_samples × 15 × 15)
    matrices_fileA[ratio] = np.mean(matrices_array, axis=0)  # Compute mean across samples

# Save mean matrices to a CSV file
with open(save_matrices_fileA, "w") as f:
    for ratio, matrix in matrices_fileA.items():
        f.write(f"Ratio: {ratio}\n")  # Write ratio label
        pd.DataFrame(matrix).to_csv(f, index=False, header=False)  # Save matrix
        f.write("\n")  # Blank line to separate matrices

print(f"Matrices fileA saved to {save_matrices_fileA}")

matrices_fileA2 = {}
# Compute mean matrix for each ratio
for ratio, matrices_list in matrixA2_sample.items():
    matrices_array = np.array(matrices_list)  # Convert list to numpy array (shape: n_samples × 15 × 15)
    matrices_fileA2[ratio] = np.mean(matrices_array, axis=0)  # Compute mean across samples

# Save mean matrices to a CSV file
with open(save_matrices_fileA2, "w") as f:
    for ratio, matrix in matrices_fileA2.items():
        f.write(f"Ratio: {ratio}\n")  # Write ratio label
        pd.DataFrame(matrix).to_csv(f, index=False, header=False)  # Save matrix
        f.write("\n")  # Blank line to separate matrices

print(f"Matrices fileA2 saved to {save_matrices_fileA2}")

#print(filenames)