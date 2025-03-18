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
ratio_threshold = 0.8
ratios = [0.8,0.7,0.6,0.5,0.4,0.3,0.2]
num_resubsampling = 10 # Number of times the training set creation is repeated to reduce standard deviation caused by individual random removal process
equal_influence_on_mean = True # If True, the mean error of the ratioed dataset is appended to the error list (from which the mean is taken at the end)

# Exclude files for which you cant get trainingsets for all ratios
#skip_files = ['data\MRT1/processed_csv_no_con\\11228_61.csv', 'data\MRT3/processed_csv_no_con\\12600_270.csv']

#files_ratio80 = ['data\\MRT1/processed_csv_no_con\\11228_15.csv', 'data\\MRT1/processed_csv_no_con\\11228_19.csv', 'data\\MRT1/processed_csv_no_con\\11228_28.csv', 'data\\MRT1/processed_csv_no_con\\11228_34.csv', 'data\\MRT1/processed_csv_no_con\\11228_35.csv', 'data\\MRT1/processed_csv_no_con\\11228_52.csv', 'data\\MRT2/processed_csv_no_con\\12600_30.csv', 'data\\MRT2/processed_csv_no_con\\12600_32.csv', 'data\\MRT2/processed_csv_no_con\\12600_64.csv', 'data\\MRT2/processed_csv_no_con\\12600_65.csv', 'data\\MRT3/processed_csv_no_con\\12600_218.csv', 'data\\MRT3/processed_csv_no_con\\12600_221.csv', 'data\\MRT3/processed_csv_no_con\\12600_228.csv', 'data\\MRT3/processed_csv_no_con\\12600_239.csv', 'data\\MRT3/processed_csv_no_con\\12600_241.csv', 'data\\MRT3/processed_csv_no_con\\12600_261.csv']
files_ratio80_split85 = ['data\\MRT1/processed_csv_no_con\\11228_28.csv', 'data\\MRT1/processed_csv_no_con\\11228_34.csv', 'data\\MRT1/processed_csv_no_con\\11228_35.csv', 'data\\MRT1/processed_csv_no_con\\11228_52.csv', 'data\\MRT2/processed_csv_no_con\\12600_25.csv', 'data\\MRT2/processed_csv_no_con\\12600_30.csv', 'data\\MRT2/processed_csv_no_con\\12600_32.csv', 'data\\MRT3/processed_csv_no_con\\12600_221.csv', 'data\\MRT3/processed_csv_no_con\\12600_239.csv', 'data\\MRT3/processed_csv_no_con\\12600_241.csv']
# Set the filename of the results
filename = f'mae_{num_resubsampling}ratio80_splits{len(splits)}_scatter.csv'
save_path = os.path.join("results_ratio", filename)

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

# Lists of dictionaries to store ratios and corresponding errors for each participant
filenames = []
results_mae_dicts = []
results_mac_dicts = []

dataset_list, files = utils.load_dataset(data_folder, subfolders, emas, emis, centered=True)

files_ratio = []
num_analysed_files = 0
for idx, dataset in enumerate(dataset_list):

    if files[idx] not in files_ratio80_split85:
        continue
    X, U = dataset['X'], dataset['Inp']

    if len(X) < num_rows_threshold:
        continue

    results = {ratio: [] for ratio in ratios}
    results_mac = {ratio: [] for ratio in ratios}

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

        files_ratio.append(files[idx])

        num_analysed_files += 1
        print(f'Loop {num_analysed_files}/16')
        print(files[idx])

        #print(f'Valid ratio train {valid_train_ratio}')
        for ratio in ratios:
            if valid_train_ratio < ratio:
                continue
            # Do multiple ratioed dataset creations -> multiple different removel processes -> decrease standard deviation caused by individual random removal process

            mae_per_sample = []
            mac_per_sample = []
            for i in range(num_resubsampling):
                resulting_training_set = get_training_set(valid_train_ratio, ratio, X_train, U_train)
                if resulting_training_set == False:
                    print(files[idx])
                    print("resulting_training_set == False")
                    break
                X_train_ratio, U_train_ratio = resulting_training_set
                mean_mae_ratio = prediction_error(X_train_ratio, U_train_ratio, X_test, U_test)
                mean_mac_ratio = mac(X_train_ratio)
                mae_per_sample.append(mean_mae_ratio)
                mac_per_sample.append(mean_mac_ratio)
            
            results[ratio].append(sum(mae_per_sample) / len(mae_per_sample))
            results_mac[ratio].append(sum(mac_per_sample) / len(mac_per_sample))
            #print(f'Valid ratio: {ratio} real valid ratio {get_valid_ratio(X_train_ratio)}; MAE: {sum(mae_per_sample) / len(mae_per_sample)} MAC: {sum(mac_per_sample) / len(mac_per_sample)}')
    filenames.append(files[idx])
    results_mae_dicts.append(results)
    results_mac_dicts.append(results_mac)

# Set up colors based on participants (files)
colors = plt.cm.get_cmap('tab10', len(files_ratio80_split85))  # Use a colormap for different participants

# Prepare data for plotting
x_values = []  # List to hold the ratios
y_values = []  # List to hold the MAE values
color_values = []  # List to hold the color values for participants
labels = []  # List to hold the labels (file names)

csv_data = []  # List to store data for CSV file

extracted_filenames = []
for file in files_ratio80_split85:
    # Extract the required part (e.g. MRT1-11228_28)
    # Extract the MRT number from the file path
    mrt_number = file.split('\\')[1].split('/')[0]
    # Extract the file name without extension
    file_name = os.path.basename(file).split('.')[0] 
    extracted_part = f"{mrt_number}-{file_name}"
    extracted_filenames.append(extracted_part)

# Iterate through each file (participant)
for i, results in enumerate(results_mae_dicts):
    print(files_ratio80_split85[i])
    for ratio in ratios:
        if ratio in results:
            # Add the ratio, MAE, and participant color
            x_values.append(ratio)
            y_values.append(np.mean(results[ratio]))  # Use mean MAE for each ratio
            color_values.append(colors(i))  # Assign color based on participant
            labels.append(files_ratio80_split85[i].split('\\')[-1])  # Extract the file name as label

            # Store data for CSV (Ratio, Mean MAE, File)
            csv_data.append([ratio, np.mean(results[ratio]), extracted_filenames[i]])
            print(f'ratio {ratio}: mean: {np.mean(results[ratio])}')


# Save results to CSV
df_results = pd.DataFrame(csv_data, columns=["Ratio", "Mean MAE", "File"])
df_results.to_csv(save_path, index=False)  # Save to CSV file

# Plot the scatter plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(x_values, y_values, c=color_values, label='Participants', alpha=1)

# Add legend with file labels
handles, _ = scatter.legend_elements()
plt.legend(handles, labels, title="Participants", bbox_to_anchor=(1.05, 1), loc='upper left')

# Labeling
plt.xlabel('Ratio')
plt.ylabel('Mean MAE')
plt.title('Scatter Plot of Mean MAE for Different Ratios')
plt.grid(True)
plt.colorbar(label='Participants')  # Colorbar to indicate different participants
plt.show()

"""
# Compute mean and standard deviation for each ratio

mean_differences = [np.mean(differences) for differences in results_mac.values()]

print(f'Datasets analysed {num_analysed_files}')
# Compute mean, standard deviation, standard errors for each ratio
def compute_me_sd_se(results_dict):
    mean_errors = [np.mean(errors) for errors in results_dict.values()]
    sd_errors = [np.std(errors) for errors in results_dict.values()]
    # Compute std error
    num_elements = [len(errors) for errors in results_dict.values()]
    se_errors = [0] * len(ratios)  # Initialize se_errors as a list of zeros
    for i,_ in enumerate(ratios):
        print(f'Number of samples for valid ratio {ratios[i]}: {num_elements[i]}')
        se_errors[i] = sd_errors[i] / np.sqrt(num_elements[i])
    return mean_errors, sd_errors, se_errors

mean_errors, sd_errors, se_errors = compute_me_sd_se(results)

# Convert results to a DataFrame to save them to a csv file
df_final = pd.DataFrame({
    "ratio": ratios,
    "mean_error": mean_errors,
    "std_dev": sd_errors,
    "std_error": se_errors,
    "mean_differences": mean_differences
})
# Save to CSV
filepath = os.path.join("results_ratio", filename)
df_final.to_csv(filepath, index=False)
print(f'Final results saved to {filename}')

# Reverse the data
ratios_reversed = ratios[::-1]
mean_errors_reversed = mean_errors[::-1]
sd_errors_reversed = sd_errors[::-1]
se_errors_reversed = se_errors[::1]
mean_differences_reversed = mean_differences[::-1]

print(files_ratio)

# Create the plot
plt.figure(figsize=(8, 6))
plt.errorbar(ratios_reversed, mean_errors_reversed, yerr=se_errors_reversed, fmt='-o', capsize=5, label='Mean Error ± Std Error')
plt.plot(ratios_reversed, mean_differences_reversed, label='mac')
# Customize the plot
plt.title('Mean Error vs Ratio with Standard Error')
plt.xlabel('Ratios')
plt.ylabel('Mean Error')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()
"""