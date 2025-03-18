import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ctrl import discrete_optimal_control as doc
from ctrl import utils
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt


emas = ['EMA_mood', 'EMA_disappointed', 'EMA_scared', 'EMA_worry', 'EMA_down', 'EMA_sad', 'EMA_confidence', 'EMA_stress', 'EMA_lonely', 'EMA_energetic', 'EMA_concentration', 'EMA_resilience', 'EMA_tired', 'EMA_satisfied', 'EMA_relaxed']
emis = ['interactive1', 'interactive2', 'interactive3', 'interactive4', 'interactive5', 'interactive6', 'interactive7', 'interactive8']

data_folder = "data"
subfolders = ["MRT1/processed_csv_no_con","MRT2/processed_csv_no_con","MRT3/processed_csv_no_con"]

# Set parameters
split = 0.7
num_rows_threshold = 50 # One file is excluded
ratios = [0.8,0.7,0.6,0.5,0.4,0.3,0.2]
num_resubsampling = 10 # Number of times the training set creation is repeated to reduce standard deviation caused by individual random removal process

files_ratio80 = ['data\\MRT1/processed_csv_no_con\\11228_15.csv', 'data\\MRT1/processed_csv_no_con\\11228_19.csv', 'data\\MRT1/processed_csv_no_con\\11228_28.csv', 'data\\MRT1/processed_csv_no_con\\11228_34.csv', 'data\\MRT1/processed_csv_no_con\\11228_35.csv', 'data\\MRT1/processed_csv_no_con\\11228_52.csv', 'data\\MRT2/processed_csv_no_con\\12600_30.csv', 'data\\MRT2/processed_csv_no_con\\12600_32.csv', 'data\\MRT2/processed_csv_no_con\\12600_64.csv', 'data\\MRT2/processed_csv_no_con\\12600_65.csv', 'data\\MRT3/processed_csv_no_con\\12600_218.csv', 'data\\MRT3/processed_csv_no_con\\12600_221.csv', 'data\\MRT3/processed_csv_no_con\\12600_228.csv', 'data\\MRT3/processed_csv_no_con\\12600_239.csv', 'data\\MRT3/processed_csv_no_con\\12600_241.csv', 'data\\MRT3/processed_csv_no_con\\12600_261.csv']
# Set the filenames of the results
filename_A = f'frobeniusA{num_resubsampling}ratio80.csv'
filename_K = f'frobeniusK{num_resubsampling}ratio80.csv'
filename_AC = f'frobeniusAC{num_resubsampling}ratio80.csv'

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
    if abs(ratio_last - target_ratio) > 0.05:
        return False
    else:
        return X_last, U_last
    
def locf(X_train): 
    df_helper_locf = pd.DataFrame(X_train).copy()
    df_helper_locf.ffill(inplace=True)
    X_train_locf = df_helper_locf.to_numpy()
    return X_train_locf
    
def frobenius_norm(matrix):
    return np.sqrt(np.sum(np.abs(matrix) ** 2))

def norm_per_ratio(X_train_ratio, U_train_ratio):
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

# Dictionaries to store ratios and corresponding frobenius norms
results_A = {ratio: [] for ratio in ratios}
results_K = {ratio: [] for ratio in ratios}
results_AC = {ratio: [] for ratio in ratios}

dataset_list = []
files = []
for subfolder in subfolders:
    folder_path = os.path.join(data_folder, subfolder, "*.csv")
    for file in glob.glob(folder_path):
        df = pd.read_csv(file)
        data = utils.csv_to_dataset(file, emas, emis, invert_columns=[])
        dataset_list.append(data)
        files.append(file)

num_analysed_files = 0
for idx, dataset in enumerate(dataset_list):
    if files[idx] not in files_ratio80:
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
        normsA_resamp = []
        normsK_resamp = []
        normsAC_resamp = []
        for i in range(num_resubsampling):
            resulting_training_set = get_training_set(valid_train_ratio, ratio, X_train, U_train)
            if resulting_training_set == False:
                continue
            X_train_ratio, U_train_ratio = resulting_training_set
            current_ratio = get_valid_ratio(X_train_ratio)  # Update current_ratio
            frobenius_norm_A, frobenius_norm_K, l2_norm_AC = norm_per_ratio(X_train_ratio, U_train_ratio)
            normsA_resamp.append(frobenius_norm_A)
            normsK_resamp.append(frobenius_norm_K)
            normsAC_resamp.append(l2_norm_AC)
        
        if normsA_resamp != []:
            results_A[ratio].append(sum(normsA_resamp) / len(normsA_resamp))
            results_K[ratio].append(sum(normsK_resamp) / len(normsK_resamp))
            results_AC[ratio].append(sum(normsAC_resamp) / len(normsAC_resamp))
            #print(f'Valid ratio: {ratio} real valid ratio {current_ratio}')

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
    

# Compute mean and standard deviation for each ratio
mean_norms_A, sd_A, se_A = compute_me_sd_se(results_A)

mean_norms_K, sd_K, se_K = compute_me_sd_se(results_K)

mean_norms_AC, sd_AC, se_AC = compute_me_sd_se(results_AC)

# Convert results to a DataFrame to save them to a csv file
folder = "results_ratio"
df_A = pd.DataFrame({
    "ratio": ratios,
    "mean_norms": mean_norms_A,
    "sd": sd_A,
    "se": se_A
})
filepath = os.path.join(folder, filename_A)
df_A.to_csv(filepath, index=False)
print(f'Final results saved to {filename_A}')

df_K = pd.DataFrame({
    "ratio": ratios,
    "mean_norms": mean_norms_K,
    "sd": sd_K,
    "se": se_K
})
# Save to CSV
filepath = os.path.join(folder, filename_K)
df_K.to_csv(filepath, index=False)
print(f'Final results saved to {filename_K}')

df_AC = pd.DataFrame({
    "ratio": ratios,
    "mean_norms": mean_norms_AC,
    "sd": sd_AC,
    "se": se_AC
})
# Save to CSV 
filepath = os.path.join(folder, filename_AC)
df_AC.to_csv(filepath, index=False)
print(f'Final results saved to {filename_AC}')


num_elements = [len(errors) for errors in results_A.values()]
print(f'Datasets analysed {num_analysed_files}')
for i,_ in enumerate(ratios):
    print(f'Number of samples for valid ratio {ratios[i]}: {num_elements[i]}')

# Reverse the data
ratios_reversed = ratios[::-1]
mean_norms_reversed = mean_norms_A[::-1]
std_errs_reversed = se_A[::-1]

# Create the plot
plt.figure(figsize=(8, 6))
plt.errorbar(ratios_reversed, mean_norms_reversed, yerr=std_errs_reversed, fmt='-o', capsize=5, label='Frobenius A')

# Customize the plot
plt.title('Frobenius A vs Ratio')
plt.xlabel('Ratios')
plt.ylabel('Frobenius')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()

mean_norms_reversed = mean_norms_K[::-1]
std_errs_reversed = se_K[::-1]

# Create the plot
plt.figure(figsize=(8, 6))
plt.errorbar(ratios_reversed, mean_norms_reversed, yerr=std_errs_reversed, fmt='-o', capsize=5, label='Frobenius K')

# Customize the plot
plt.title('Frobenius K vs Ratio')
plt.xlabel('Ratios')
plt.ylabel('Frobenius')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()

mean_norms_reversed = mean_norms_AC[::-1]
std_errs_reversed = se_AC[::-1]

# Create the plot
plt.figure(figsize=(8, 6))
plt.errorbar(ratios_reversed, mean_norms_reversed, yerr=std_errs_reversed, fmt='-o', capsize=5, label='Frobenius AC')

# Customize the plot
plt.title('Frobenius AC vs Ratio')
plt.xlabel('Ratios')
plt.ylabel('Frobenius')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()