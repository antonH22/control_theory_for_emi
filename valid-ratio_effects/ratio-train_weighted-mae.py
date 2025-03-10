import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ctrl import discrete_optimal_control as doc
from ctrl import utils
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Approach: For each dataset the split index is the same, but the training set is adapted to the ratios (while the test set stays the same)
# Problems of this evaluation approach: 
# - The number of participants with valid ratio >= 80% is only 18, should you use smoothed data?
# - The ratio of valid rows in training data and testing data changes for different ratios

# Error computation: Weighted mean error (to reduce the variance caused by datasets with few valid rows)
# Problem: The participant have different influences on the error (less valid rows -> less influence)

emas = ['EMA_mood', 'EMA_disappointed', 'EMA_scared', 'EMA_worry', 'EMA_down', 'EMA_sad', 'EMA_confidence', 'EMA_stress', 'EMA_lonely', 'EMA_energetic', 'EMA_concentration', 'EMA_resilience', 'EMA_tired', 'EMA_satisfied', 'EMA_relaxed']
emis = ['interactive1', 'interactive2', 'interactive3', 'interactive4', 'interactive5', 'interactive6', 'interactive7', 'interactive8']

data_folder = "data"
subfolders = ["MRT1/processed_csv_no_con","MRT2/processed_csv_no_con","MRT3/processed_csv_no_con"]

# Set parameters
split = 0.7
num_rows_threshold = 50 # One file is excluded
ratios = [0.8,0.7,0.6,0.5,0.4,0.3,0.2]
num_resubsampling = 50 # Number of times the training set creation is repeated to reduce standard deviation caused by individual random removal process

# Set the filename of the results
filename = f'trainratio-mae{num_resubsampling}_we.csv'

def remove_one_valid_row(data, input):
    # Find indices of rows that do not contain NaN values and delete one randomly
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

def get_training_set(valid_train_ratio, target_ratio, X_train, U_train):
    # Initialize current state with the original training sets
    X_current, U_current = X_train, U_train
    current_ratio = valid_train_ratio

    # Initialize last valid state as the starting state
    X_last, U_last, ratio_last = X_train, U_train, valid_train_ratio

    while current_ratio > target_ratio:
        X_current, U_current = remove_one_valid_row(X_current, U_current)
        current_ratio = get_valid_ratio(X_current)
        # Update last valid state if the new ratio is closer to target_ratio
        if abs(current_ratio - target_ratio) < abs(ratio_last - target_ratio):
            X_last, U_last, ratio_last = X_current, U_current, current_ratio

    # Check if the best ratio difference is within the acceptable threshold
    if abs(ratio_last - target_ratio) > 0.01:
        return False
    else:
        return X_last, U_last

def prediction_error(X_train_ratio, U_train_ratio, X_test, U_test):
    A, B, lmbda = utils.stable_ridge_regression(X_train_ratio, U_train_ratio)
    mae_per_step_list = []
    
    for i in range(len(X_test) -1):
        if np.isnan(X_test[i]).any() or np.isnan(X_test[i + 1]).any():
            continue
        x_next = doc.step(A, B, X_test[i], U_test[i])
        mae_per_step = np.mean(np.abs(x_next - X_test[i+1]))
        mae_per_step_list.append(mae_per_step)

    return mae_per_step_list

# Dictionary to store ratios and corresponding errors
results = {ratio: [] for ratio in ratios}

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

    num_analysed_files += 1
    print(f'Loop {num_analysed_files}/176')
    #print('-' * 40)
    #print(f'{files[idx]}:')
    #print(f'Split index: {split_index}')
    
    # Split data
    X_train, X_test = X[:split_index], X[split_index:]
    U_train, U_test = U[:split_index], U[split_index:]

    valid_train_ratio = get_valid_ratio(X_train)

    #print(f'Valid ratio train {valid_train_ratio}')
    for ratio in ratios:
        if valid_train_ratio < ratio:
            continue
        # Do multiple ratioed dataset creations -> multiple different removel processes -> decrease standard deviation caused by individual random removal process
        mae_per_step_overall_list = []
        for i in range(num_resubsampling):
            resulting_training_set = get_training_set(valid_train_ratio, ratio, X_train, U_train)
            if resulting_training_set == False:
                mae_step_overall_list = []
                break
            X_train_ratio, U_train_ratio = resulting_training_set
            current_ratio = get_valid_ratio(X_train_ratio)  # Update current_ratio
            mae_per_step_list = prediction_error(X_train_ratio, U_train_ratio, X_test, U_test)
            mae_per_step_overall_list.extend(mae_per_step_list)
                
        if mae_per_step_overall_list != []:
            mae_per_step_overall_array = np.array(mae_per_step_overall_list)
            mean_error = np.mean(mae_per_step_overall_array)  # Compute mean per dataset
            results[ratio].append((mean_error, len(mae_per_step_overall_list)))  # Store size info
            #print(f'Valid ratio: {ratio} real valid ratio {current_ratio}; MAE: {np.mean(mae_per_step_overall_list)}')

weighted_mean_errors = []
weighted_std_devs = []
num_elements = []

for ratio in ratios:
    data = results[ratio]  # List of (mean_error, size)
    
    errors = np.array([x[0] for x in data])  # Extract mean errors
    sizes = np.array([x[1] for x in data])   # Extract sizes
    num_elements.append(len(errors))
    
    total_size = np.sum(sizes)
    weights = sizes / total_size  # Normalize weights
    
    weighted_mean = np.sum(weights * errors)  # Compute weighted mean
    weighted_mean_errors.append(weighted_mean)

    # Compute weighted standard deviation
    weighted_variance = np.sum(weights * (errors - weighted_mean) ** 2)
    weighted_std = np.sqrt(weighted_variance)  # Weighted standard deviation
    weighted_std_devs.append(weighted_std)

print(f'std_errors mean: {np.mean(weighted_mean_errors)}')
print(f'Datasets analysed {num_analysed_files}')
num_elements = [len(errors) for errors in results.values()]
for i,_ in enumerate(ratios):
    print(f'Number of mean values for valid ratio {ratios[i]}: {num_elements[i]}')

# Convert results to a DataFrame to save them to a csv file
df_final = pd.DataFrame({
    "ratio": ratios,
    "mean_error": weighted_mean_errors,
    "std_dev": weighted_std_devs,
    "num_elements": num_elements
})
# Save to CSV
filepath = os.path.join("results_ratio-mae", filename)
df_final.to_csv(filepath, index=False)
print(f'Final results saved to {filename}')

# Reverse the data
ratios_reversed = ratios[::-1]
mean_errors_reversed = weighted_mean_errors[::-1]
std_devs_reversed = weighted_std_devs[::-1]

# Create the plot
plt.figure(figsize=(8, 6))
plt.errorbar(ratios_reversed, mean_errors_reversed, yerr=std_devs_reversed, fmt='-o', capsize=5, label='Mean Error ± Std Dev')

# Customize the plot
plt.title('Mean Error vs Ratio with Standard Deviation')
plt.xlabel('Ratios')
plt.ylabel('Mean Error')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()