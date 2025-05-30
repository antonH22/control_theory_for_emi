import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from enhanced_eval_ctrl_approach import myutils
from ctrl import utils

import numpy as np
import pandas as pd

### Analyzing robustness and reliability of the LDS model across valid ratios (5.1.4, Figure 3)

emas = ['EMA_mood', 'EMA_disappointed', 'EMA_scared', 'EMA_worry', 'EMA_down', 'EMA_sad', 'EMA_confidence', 'EMA_stress', 'EMA_lonely', 'EMA_energetic', 'EMA_concentration', 'EMA_resilience', 'EMA_tired', 'EMA_satisfied', 'EMA_relaxed']
emis = ['interactive1', 'interactive2', 'interactive3', 'interactive4', 'interactive5', 'interactive6', 'interactive7', 'interactive8']

data_folder = "data"
subfolders = ["MRT1/processed_csv_no_con","MRT2/processed_csv_no_con","MRT3/processed_csv_no_con"]

# Set parameters
splits = [0.7, 0.75, 0.8, 0.85]
ratio_threshold = 0.8 # For each split the valid ratio needs to be greater than 80%
ratios = [0.2, 0.3 ,0.4, 0.5, 0.6, 0.7, 0.8]
num_resubsampling = 30 # Number of times the training set creation is repeated to reduce standard deviation caused by individual random removal process

# Files with valid ratio > 0.8 in the training set across all splits
files_ratio80 = ['data\\MRT1/processed_csv_no_con\\11228_28.csv', 'data\\MRT1/processed_csv_no_con\\11228_34.csv', 'data\\MRT1/processed_csv_no_con\\11228_35.csv', 'data\\MRT1/processed_csv_no_con\\11228_52.csv', 'data\\MRT2/processed_csv_no_con\\12600_30.csv', 'data\\MRT2/processed_csv_no_con\\12600_32.csv', 'data\\MRT3/processed_csv_no_con\\12600_221.csv', 'data\\MRT3/processed_csv_no_con\\12600_239.csv', 'data\\MRT3/processed_csv_no_con\\12600_241.csv']

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

def get_training_set(valid_train_ratio, target_ratio, X_train, U_train):
    " Iteratively removes valid rows of the training data to achieve target valid data ratio (0.05 tolerance). "
    # Initialize current state with the original training sets
    X_current, U_current = X_train, U_train
    current_ratio = myutils.get_valid_ratio(X_train)

    # Initialize last valid state as the starting state
    X_last, U_last, ratio_last = X_train, U_train, valid_train_ratio

    while current_ratio > target_ratio:
        X_current, U_current = remove_one_valid_row(X_current, U_current)
        current_ratio = myutils.get_valid_ratio(X_current)
        # Update last valid state if the new ratio is closer to target_ratio
        if abs(current_ratio - target_ratio) < abs(ratio_last - target_ratio):
            X_last, U_last, ratio_last = X_current, U_current, current_ratio

    # Check if the best ratio difference is within the acceptable threshold
    if abs(ratio_last - target_ratio) > 0.05:
        print("Threshold is reached")
        return False
    else:
        return X_last, U_last

def get_A_matrix(X_train_ratio, U_train_ratio):
    X_train_ratio = myutils.locf(X_train_ratio)
    A, B, lmbda = utils.stable_ridge_regression(X_train_ratio, U_train_ratio)
    return A

###################################################################################################################

dataset_list, files = myutils.load_dataset(data_folder, subfolders, emas, emis, centered=True)

# Lists of dictionaries to store ratios and corresponding errors for each participant
filenames = []
results_mae_dicts = []
results_eigenvalue_dicts = []
results_frobeniusA_dicts = []
results_frobeniusK_dicts = []
results_frobeniusAC_dicts = []

results_matrixA_dicts = []

filenames = []
for idx, dataset in enumerate(dataset_list):

    if files[idx] not in files_ratio80:
        continue
    print(f'Iteration {len(filenames)+1}/9')
    X, U = dataset['X'], dataset['Inp']

    valid_rows = myutils.get_valid_rows(X)
    num_valid_rows = valid_rows.sum()

    results_mae = {ratio: [] for ratio in ratios}
    results_eigenvalue = {ratio: [] for ratio in ratios}
    results_frobeniusA = {ratio: [] for ratio in ratios}
    results_frobeniusK = {ratio: [] for ratio in ratios}
    results_frobeniusAC = {ratio: [] for ratio in ratios}
    results_matrixA = {ratio: [] for ratio in ratios}
    matrices = {ratio: [] for ratio in ratios}

    for split in splits:
        # Determine the split index for the training and testing data
        split_index = np.searchsorted(np.cumsum(valid_rows), num_valid_rows * split) # searches for the index in (np.cumsum(pairs)) where the cumulative sum first exceeds 70% of the valid rows
        
        # Split data
        X_train, X_test = X[:split_index], X[split_index:]
        U_train, U_test = U[:split_index], U[split_index:]

        valid_train_ratio = myutils.get_valid_ratio(X_train)
        
        for ratio in ratios:
            # Do multiple ratioed dataset creations -> multiple different removel processes -> decrease standard deviation caused by individual random removal process
            mae_per_sample = []
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
                mean_mae_ratio = myutils.prediction_error(X_train_ratio, U_train_ratio, X_test, U_test)
                mae_per_sample.append(mean_mae_ratio)
                eigenvalue_ratio = myutils.compute_dominant_eigenvalue(X_train_ratio, U_train_ratio)
                eigenvalue_per_sample.append(eigenvalue_ratio)
                frobenius_norm_A, frobenius_norm_K, l2_norm_AC = myutils.compute_model_norms(X_train_ratio, U_train_ratio)
                frobeniusA_per_sample.append(frobenius_norm_A)
                frobeniusK_per_sample.append(frobenius_norm_K)
                frobeniusAC_per_sample.append(l2_norm_AC)

                a_matrix = get_A_matrix(X_train_ratio, U_train_ratio)
                matrixA_per_sample.append(a_matrix)

            results_mae[ratio].append(sum(mae_per_sample) / len(mae_per_sample))
            results_eigenvalue[ratio].append(sum(eigenvalue_per_sample) / len(eigenvalue_per_sample))
            results_frobeniusA[ratio].append(sum(frobeniusA_per_sample) / len(frobeniusA_per_sample))
            results_frobeniusK[ratio].append(sum(frobeniusK_per_sample) / len(frobeniusK_per_sample))
            results_frobeniusAC[ratio].append(sum(frobeniusAC_per_sample) / len(frobeniusAC_per_sample))

            results_matrixA[ratio].append(np.mean(matrixA_per_sample, axis=0))
            #print(f'Valid ratio: {ratio} real valid ratio {get_valid_ratio(X_train_ratio)}; MAE: {sum(mae_per_sample) / len(mae_per_sample)})
    filenames.append(files[idx])
    results_mae_dicts.append(results_mae)
    results_eigenvalue_dicts.append(results_eigenvalue)
    results_frobeniusA_dicts.append(results_frobeniusA)
    results_frobeniusK_dicts.append(results_frobeniusK)
    results_frobeniusAC_dicts.append(results_frobeniusAC)

    results_matrixA_dicts.append(results_matrixA)

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
results_matrix_A_flattened_dfs = []
for matrix_dict in results_matrixA_dicts:
    df_matrix_A_flattened = pd.DataFrame()
    mean_dict = {}
    variance_dict = {}
    for ratio in ratios:
        mean_matrix = np.mean(matrix_dict[ratio], axis=0)
        mean_dict[ratio] = np.mean(mean_matrix)
        variance_dict[ratio] = np.var(mean_matrix)
        df_matrix_A_flattened[ratio] = mean_matrix.flatten()
    results_mean_A_mean_dicts.append(mean_dict)   
    results_var_A_mean_dicts.append(variance_dict)
    results_matrix_A_flattened_dfs.append(df_matrix_A_flattened)


# Iterate through each file (participant) and compute the means per ratio
for i, (mae_part, eigenvalue_part, frobA_part, frobK_part, frobAC_part, meanA_part, varA_part, df_flatA_part) in enumerate(zip(results_mae_dicts, results_eigenvalue_dicts, results_frobeniusA_dicts, results_frobeniusK_dicts, results_frobeniusAC_dicts, results_mean_A_mean_dicts, results_var_A_mean_dicts, results_matrix_A_flattened_dfs)):
    correlations = df_flatA_part.corrwith(df_flatA_part[0.8])
    for ratio in ratios:
        # Store data for CSV (Ratio, Mean MAE, Eigenvalue, File)
        csv_data.append([
            ratio, 
            np.mean(mae_part[ratio]),
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
save_path = os.path.join("results_replicated", 'results_valid_ratios_metrics_new.csv')
df_results = pd.DataFrame(csv_data, columns=["Ratio", "Mean MAE", "Mean Eigenvalue","Frobenius A", "Frobenius K", "Frobenius AC","Mean A","Variance A", "Correlation A", "File"])
df_results.to_csv(save_path, index=False)  # Save to CSV file
print(f'Saved all results to {save_path}')



"""
# Output matrix A (mean over all 30 subsamplings) for two files across all ratios to analyze anomaly at ratio 0.2
fileA = 'data\\MRT1/processed_csv_no_con\\11228_34.csv'
fileA2 = 'data\\MRT1/processed_csv_no_con\\11228_52.csv'
save_matrices_fileA = os.path.join("results_ratio", filename[:-4]+"_matrixA_MRT1-34.csv")
save_matrices_fileA2 = os.path.join("results_ratio", filename[:-4]+"_matrixA_MRT1-52.csv")

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
"""