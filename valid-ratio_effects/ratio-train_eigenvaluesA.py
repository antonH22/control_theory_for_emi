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
split = 0.7
num_rows_threshold = 50 # One file is excluded
num_resubsampling = 10 # Number of times the training set creation is repeated to reduce standard deviation caused by individual random removal process
ratios = [0.8,0.7,0.6,0.5,0.4,0.3,0.2]
ratio_reduction = 0.1 # When this is set to None the Eigenvalues are computed for each ratio

files_ratio80 = ['data\\MRT1/processed_csv_no_con\\11228_15.csv', 'data\\MRT1/processed_csv_no_con\\11228_19.csv', 'data\\MRT1/processed_csv_no_con\\11228_28.csv', 'data\\MRT1/processed_csv_no_con\\11228_34.csv', 'data\\MRT1/processed_csv_no_con\\11228_35.csv', 'data\\MRT1/processed_csv_no_con\\11228_52.csv', 'data\\MRT2/processed_csv_no_con\\12600_30.csv', 'data\\MRT2/processed_csv_no_con\\12600_32.csv', 'data\\MRT2/processed_csv_no_con\\12600_64.csv', 'data\\MRT2/processed_csv_no_con\\12600_65.csv', 'data\\MRT3/processed_csv_no_con\\12600_218.csv', 'data\\MRT3/processed_csv_no_con\\12600_221.csv', 'data\\MRT3/processed_csv_no_con\\12600_228.csv', 'data\\MRT3/processed_csv_no_con\\12600_239.csv', 'data\\MRT3/processed_csv_no_con\\12600_241.csv', 'data\\MRT3/processed_csv_no_con\\12600_261.csv']
# Set the filename of the results
filename = f'eigenvaluesA{num_resubsampling}ratio80_participants_testfilenames.csv'

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

# Lists to store ratios and corresponding eigenvalues
filenames = []
results_ratios = []
results_eigenvalues = []

dataset_list, files = utils.load_dataset(data_folder, subfolders, emas, emis, centered=True)

skip_files = {}
num_analysed_files = 0
for idx, dataset in enumerate(dataset_list):
    if files[idx] not in files_ratio80:
        continue
    X, U = dataset['X'], dataset['Inp']

    if len(X) < num_rows_threshold:
        continue

    results_ratios_participant = []
    results_eigenvalues_participant = []
    
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

    filenames.append(files[idx])

    if ratio_reduction != None:
        skip = False
        eigenvalues = compute_eigenvalues(X_train, U_train)
        dominant_eigenvalue = np.max(np.abs(eigenvalues))
        results_ratios_participant.append(initial_ratio)
        results_eigenvalues_participant.append(dominant_eigenvalue)
        
        #print(f'Valid ratio train {valid_train_ratio}')
        iterations = 1
        target_ratio = initial_ratio - ratio_reduction
        while target_ratio > 0.1:
            # Do multiple ratioed dataset creations -> multiple different removel processes -> decrease standard deviation caused by individual random removal process
            target_ratio = initial_ratio - ratio_reduction * iterations
            ratios_resamp = []
            eigenvalues_resamp = []
            for i in range(num_resubsampling):
                X_train_ratio, U_train_ratio, current_ratio = get_training_set(target_ratio, X_train, U_train)
                if current_ratio == 0.0:
                    skip = True
                    continue
                eigenvalues = compute_eigenvalues(X_train_ratio, U_train_ratio)
                dominant_eigenvalue = np.max(np.abs(eigenvalues))
                ratios_resamp.append(current_ratio)
                eigenvalues_resamp.append(dominant_eigenvalue)
                #print(f'Ratio: {current_ratio}; dominant eigenvalue: {dominant_eigenvalue}')
            
            if skip:
                print("Exited while loop because current_ratio == 0: new participant iteration")
                break
            mean_ratio_resamp = sum(ratios_resamp) / len(ratios_resamp)
            mean_eigenvalue_resamp = sum(eigenvalues_resamp) / len(eigenvalues_resamp)
            results_ratios_participant.append(mean_ratio_resamp)
            results_eigenvalues_participant.append(mean_eigenvalue_resamp)
        
            iterations += 1
        results_ratios.append(results_ratios_participant)
        results_eigenvalues.append(results_eigenvalues_participant)
    """"
    Todo: tranform this copied code to the eigenvalue computation
    else:
        for ratio in ratios:
            if initial_ratio < ratio:
                continue
            # Do multiple ratioed dataset creations -> multiple different removel processes -> decrease standard deviation caused by individual random removal process
            mae_per_step_overall_list = []
            mean_mac_overall_list = []
            for i in range(num_resubsampling):
                resulting_training_set = get_training_set(valid_train_ratio, ratio, X_train, U_train)
                if resulting_training_set == False:
                    mae_step_overall_list = []
                    print(files[idx])
                    print("resulting_training_set == False")
                    break
                X_train_ratio, U_train_ratio = resulting_training_set
                mae_per_step_list = prediction_error(X_train_ratio, U_train_ratio, X_test, U_test)
                mean_mac = mac(X_test)
                mean_mac_overall_list.append(mean_mac)
                if equal_influence_on_mean:
                    mae_per_step_overall_list.append(mae_per_step_list)
                else:
                    mae_per_step_overall_list.extend(mae_per_step_list)
                    
            if mae_per_step_overall_list != []:
                mae_per_step_overall_array = np.array(mae_per_step_overall_list)
                #mae_per_step_overall_mean = np.mean(mae_per_step_overall_array, axis = 0)
                mae_per_step_overall_mean = np.mean(mae_per_step_overall_array)

                mac_overall_mean = sum(mean_mac_overall_list) / len(mean_mac_overall_list)
                results_mac[ratio].append(mac_overall_mean)
                # Step 7: Store the ratio and corresponding error
                if equal_influence_on_mean:
                    results[ratio].append(mae_per_step_overall_mean)
                else:
                    results[ratio].extend(mae_per_step_overall_list)
                #print(f'Valid ratio: {ratio} real valid ratio {get_valid_ratio(X_train_ratio)}; MAE: {np.mean(mae_per_step_overall_list)} MAC: {mac_overall_mean}')
    """

# Set up colors based on participants (files)
colors = plt.cm.get_cmap('tab10', len(results_ratios))  # Use a colormap for different participants

extracted_filenames = []

for file in filenames:
    # Extract the required part (e.g. MRT1-11228_28)
    # Extract the MRT number from the file path
    mrt_number = file.split('\\')[1].split('/')[0]
    # Extract the file name without extension
    file_name = os.path.basename(file).split('.')[0] 
    extracted_part = f"{mrt_number}-{file_name}"
    extracted_filenames.append(extracted_part)

# Convert results to a DataFrame to save them to a csv file
df_final = pd.DataFrame({
    "ratio": results_ratios,
    "eigenvalues": results_eigenvalues,
    "filenames": extracted_filenames
})

# Save to CSV
filepath = os.path.join("results_ratio", filename)
df_final.to_csv(filepath, index=False)
print(f'Final results saved to {filename}')

print(f'Datasets analysed {num_analysed_files}')
print(f'Number of datapoints: {len(results_ratios)}')

# Reverse the data
ratios_reversed = results_ratios[::-1]
dominant_eigenvalues_reversed = results_eigenvalues[::-1]

# Create scatter plot: valid data ratio vs. dominant eigenvalue magnitude
plt.figure(figsize=(8, 6))
plt.scatter(ratios_reversed, dominant_eigenvalues_reversed, color='b', label='Dominant Eigenvalue')
plt.xlabel('Valid Data Ratio')
plt.ylabel('Dominant Eigenvalue Magnitude')
plt.title('Dominant Eigenvalue vs. Valid Data Ratio')
plt.grid(True)
plt.legend()
plt.show()