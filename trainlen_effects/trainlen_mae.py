import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ctrl import discrete_optimal_control as doc
from ctrl import utils
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Approach: For each dataset the split index is the same, but the training set is adapted to the train set lengths by removing rows in the beginning (while the test set stays the same)

# Error computation: Weighted mean error 
# Problem: The participant have different influences on the error (less valid rows -> less influence)

emas = ['EMA_mood', 'EMA_disappointed', 'EMA_scared', 'EMA_worry', 'EMA_down', 'EMA_sad', 'EMA_confidence', 'EMA_stress', 'EMA_lonely', 'EMA_energetic', 'EMA_concentration', 'EMA_resilience', 'EMA_tired', 'EMA_satisfied', 'EMA_relaxed']
emis = ['interactive1', 'interactive2', 'interactive3', 'interactive4', 'interactive5', 'interactive6', 'interactive7', 'interactive8']

data_folder = "data"
subfolders = ["MRT1/processed_csv_no_con","MRT2/processed_csv_no_con","MRT3/processed_csv_no_con"]

# Set parameters
split = 0.7 # Not on basis of the valid ratio, but total length (different from the ratio analysis scripts)
num_rows_threshold = 50 # One file is excluded
trainlens = [160,140,120,100,80,60,40,20]

# Set the filename of the results
filename = f'trainlen-mae.csv'

def get_training_set(trainlen, X_train, U_train):
    number_to_remove = len(X_train) - trainlen
    if number_to_remove < 0:
        return False
    X_train_new = X_train[number_to_remove:]
    U_train_new = U_train[number_to_remove:]
    return X_train_new, U_train_new

def locf(X_train): 
    df_helper_locf = pd.DataFrame(X_train).copy()
    df_helper_locf.ffill(inplace=True)
    X_train_locf = df_helper_locf.to_numpy()
    return X_train_locf

def prediction_error(X_train_trainlen, U_train_trainlen, X_test, U_test):
    X_train_trainlen = locf(X_train_trainlen)
    A, B, lmbda = utils.stable_ridge_regression(X_train_trainlen, U_train_trainlen)
    mae_per_step_list = []
    
    for i in range(len(X_test) -1):
        if np.isnan(X_test[i]).any() or np.isnan(X_test[i + 1]).any():
            continue
        x_next = doc.step(A, B, X_test[i], U_test[i])
        mae_per_step = np.mean(np.abs(x_next - X_test[i+1]))
        mae_per_step_list.append(mae_per_step)

    return mae_per_step_list

# Dictionary to store train set length and corresponding errors
results = {trainlens: [] for trainlens in trainlens}

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
    split_index = int(np.floor(len(X) * 0.7))

    num_analysed_files += 1
    print(f'Loop {num_analysed_files}/176')
    #print('-' * 40)
    print(f'{files[idx]}:')
    #print(f'Split index: {split_index}')
    
    # Split data
    X_train, X_test = X[:split_index], X[split_index:]
    U_train, U_test = U[:split_index], U[split_index:]

    if len(X_train) < trainlens[0]:
        continue

    for trainlen in trainlens:
        if len(X_train) < trainlen:
            continue
        mae_per_step_overall_list = []
        resulting_training_set = get_training_set(trainlen, X_train, U_train)
        if resulting_training_set == False:
            mae_step_overall_list = []
            break
        X_train_trainlen, U_train_trainlen = resulting_training_set
        mae_per_step_list = prediction_error(X_train_trainlen, U_train_trainlen, X_test, U_test)
        mae_per_step_overall_list.extend(mae_per_step_list)
                
        if mae_per_step_overall_list != []:
            mae_per_step_overall_array = np.array(mae_per_step_overall_list)
            mean_error = np.mean(mae_per_step_overall_array)  # Compute mean per dataset
            results[trainlen].append(mean_error)  # Store size info
            print(f'Valid trainlen: {trainlen}, real trainlen {len(X_train_trainlen)}; MAE: {mean_error}')

# Compute mean, standard deviation, standard errors for each ratio
def compute_me_sd_se(results_dict):
    mean_errors = [np.mean(errors) for errors in results_dict.values()]
    sd_errors = [np.std(errors) for errors in results_dict.values()]
    # Compute std error
    num_elements = [len(errors) for errors in results_dict.values()]
    se_errors = [0] * len(trainlens)  # Initialize se_errors as a list of zeros
    for i,_ in enumerate(trainlens):
        print(f'Number of samples for valid ratio {trainlens[i]}: {num_elements[i]}')
        se_errors[i] = sd_errors[i] / np.sqrt(num_elements[i])
    return mean_errors, sd_errors, se_errors

mean_errors, sd_errors, se_errors = compute_me_sd_se(results)

# Convert results to a DataFrame to save them to a csv file
df_final = pd.DataFrame({
    "trainlen": trainlens,
    "mean_error": mean_errors,
    "std_dev": sd_errors,
    "std_error": se_errors
})
# Save to CSV
filepath = os.path.join("results_trainlen", filename)
df_final.to_csv(filepath, index=False)
print(f'Final results saved to {filename}')

# Reverse the data
trainlens_reversed = trainlens[::-1]
mean_errors_reversed = mean_errors[::-1]
std_errs_reversed = se_errors[::-1]

# Create the plot
plt.figure(figsize=(8, 6))
plt.errorbar(trainlens_reversed, mean_errors_reversed, yerr=std_errs_reversed, fmt='-o', capsize=5, label='Mean Error ± Std Dev')

# Customize the plot
plt.title('Mean Error vs Ratio with Standard Deviation')
plt.xlabel('Ratios')
plt.ylabel('Mean Error')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()