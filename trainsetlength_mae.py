from ctrl import discrete_optimal_control as doc
from ctrl import utils
from investigate_missing_data import compute_missing_data_percentage
import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


emas = ['EMA_mood', 'EMA_disappointed', 'EMA_scared', 'EMA_worry', 'EMA_down', 'EMA_sad', 'EMA_confidence', 'EMA_stress', 'EMA_lonely', 'EMA_energetic', 'EMA_concentration', 'EMA_resilience', 'EMA_tired', 'EMA_satisfied', 'EMA_relaxed']
emis = ['interactive1', 'interactive2', 'interactive3', 'interactive4', 'interactive5', 'interactive6', 'interactive7', 'interactive8']

data_folder = "data"
subfolders = ["MRT1/processed_csv_no_con","MRT2/processed_csv_no_con","MRT3/processed_csv_no_con"]

# Set the split ratio
split = 0.7
split_index_threshold = 0
num_rows_threshold = 50 # One file is excluded
train_set_lengths = [100,90,80,70,60,50,40,30,20]
num_resubsampling = 1 # Number of times the training set creation is repeated to reduce standard deviation caused by individual random removal process

# Set the filename of the results
filename = "trainlen-mae1.csv"

def get_training_set(train_set_length, X_train, U_train):
    number_to_remove = len(X_train) - train_set_length
    non_nan_rows = np.where(~np.isnan(X_train).any(axis=1))[0] 
    if number_to_remove > len(non_nan_rows):
        return False
    # Randomly select n rows to remove
    rows_to_remove = np.random.choice(non_nan_rows, size=number_to_remove, replace=False)
    # Delete selected rows from both arrays
    X_train = np.delete(X_train, rows_to_remove, axis=0)
    U_train = np.delete(U_train, rows_to_remove, axis=0)
    return X_train, U_train

def prediction_error(X_train, U_train, X_test, U_test):
    A, B, lmbda = utils.stable_ridge_regression(X_train, U_train)
    mae_per_step_list = []
    
    for i in range(len(X_test) -1):
        if np.isnan(X_test[i]).any() or np.isnan(X_test[i + 1]).any():
            continue
        x_next = doc.step(A, B, X_test[i], U_test[i])
        mae_per_step = np.mean(np.abs(x_next - X_test[i+1]))
        mae_per_step_list.append(mae_per_step)

    return mae_per_step_list

# Dictionary to store train set length and corresponding errors
results = {train_set_length: [] for train_set_length in train_set_lengths}

dataset_list = []
files = []
for subfolder in subfolders:
    folder_path = os.path.join(data_folder, subfolder, "*.csv")
    for file in glob.glob(folder_path):
        df = pd.read_csv(file)
        data = utils.csv_to_dataset(file, emas, emis, invert_columns=[])
        dataset_list.append(data)
        files.append(file)

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

    if split_index < split_index_threshold:
        continue

    num_analysed_files += 1
    print(f'Loop {num_analysed_files}/176')
    #print('-' * 40)
    #print(f'{files[idx]}:')
    #print(f'Split index: {split_index}')
    
    # Split data
    X_train, X_test = X[:split_index], X[split_index:]
    U_train, U_test = U[:split_index], U[split_index:]

    #print(f'Valid ratio train {valid_train_ratio}')
    for train_set_length in train_set_lengths:
        if len(X_train) < train_set_length:
            continue
        # Do multiple ratioed dataset creations -> multiple different removel processes -> decrease standard deviation caused by individual random removal process
        mae_per_step_overall_list = []
        for i in range(num_resubsampling):
            resulting_training_set = get_training_set(train_set_length, X_train, U_train)
            if resulting_training_set == False:
                mae_step_overall_list = []
                break
            X_train, U_train = resulting_training_set
            mae_per_step_list = prediction_error(X_train, U_train, X_test, U_test)
            mae_per_step_overall_list.append(mae_per_step_list)
                
        if mae_per_step_overall_list != []:
            mae_per_step_overall_array = np.array(mae_per_step_overall_list)
            mae_per_step_overall_mean = np.mean(mae_per_step_overall_array, axis = 0)
            # Step 7: Store the ratio and corresponding error
            results[train_set_length].extend(mae_per_step_overall_mean)
            #print(f'Train set length: {train_set_length} real length {len(X_train)}; MAE: {np.mean(mae_per_step_overall_list)}')
    

# Compute mean and standard deviation for each ratio
mean_errors = [np.mean(errors) for errors in results.values()]
std_errors = [np.std(errors) for errors in results.values()]
print(f'std_errors mean: {np.mean(std_errors)}')

# Convert results to a DataFrame to save them to a csv file
df_final = pd.DataFrame({
    "train_set_length": train_set_lengths,
    "mean_error": mean_errors,
    "std_dev": std_errors
})
# Save to CSV
filepath = os.path.join("results_trainlen-mae", filename)
df_final.to_csv(filepath, index=False)
print(f'Final results saved to {filename}')

num_elements = [len(errors) for errors in results.values()]
print(f'Datasets analysed {num_analysed_files}')
for i,_ in enumerate(train_set_lengths):
    print(f'Number of samples for valid ratio {train_set_lengths[i]}: {num_elements[i]}')

# Reverse the data
reversed_len = train_set_lengths[::-1]
reversed_mae = mean_errors[::-1]
reversed_std = std_errors[::-1]

# Create the plot
plt.figure(figsize=(8, 6))
plt.errorbar(reversed_len, reversed_mae, yerr=reversed_std, fmt='-o', capsize=5, label='Mean Error ± Std Dev')

# Customize the plot
plt.title('Mean Error vs Ratio with Standard Deviation')
plt.xlabel('Ratios')
plt.ylabel('Mean Error')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()