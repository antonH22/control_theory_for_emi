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
split = 0.7 # Not on basis of the valid ratio, but total length (different from the ratio analysis scripts)
num_rows_threshold = 50 # One file is excluded
trainlens = [160,140,120,100,80,60,40,20]

#files_ratio80 = ['data\\MRT1/processed_csv_no_con\\11228_15.csv', 'data\\MRT1/processed_csv_no_con\\11228_19.csv', 'data\\MRT1/processed_csv_no_con\\11228_28.csv', 'data\\MRT1/processed_csv_no_con\\11228_34.csv', 'data\\MRT1/processed_csv_no_con\\11228_35.csv', 'data\\MRT1/processed_csv_no_con\\11228_52.csv', 'data\\MRT2/processed_csv_no_con\\12600_30.csv', 'data\\MRT2/processed_csv_no_con\\12600_32.csv', 'data\\MRT2/processed_csv_no_con\\12600_64.csv', 'data\\MRT2/processed_csv_no_con\\12600_65.csv', 'data\\MRT3/processed_csv_no_con\\12600_218.csv', 'data\\MRT3/processed_csv_no_con\\12600_221.csv', 'data\\MRT3/processed_csv_no_con\\12600_228.csv', 'data\\MRT3/processed_csv_no_con\\12600_239.csv', 'data\\MRT3/processed_csv_no_con\\12600_241.csv', 'data\\MRT3/processed_csv_no_con\\12600_261.csv']

# Set the filenames of the results
filename_A = f'trainlen_frobenius_A.csv'
filename_K = f'trainlen_frobenius_K.csv'
filename_AC = f'trainlen_frobenius_AC.csv'

def get_training_set(trainlen, X_train, U_train):
    number_to_remove = len(X_train) - trainlen
    if number_to_remove < 0:
        return False
    X_train_new = X_train[number_to_remove:]
    U_train_new = U_train[number_to_remove:]
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
    X_train_trainlen = locf(X_train_trainlen)
    A, B, lmbda = utils.stable_ridge_regression(X_train_trainlen, U_train_trainlen)
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

dataset_list, files = utils.load_dataset(data_folder, subfolders, emas, emis, centered=True)

skip_files = {}
num_analysed_files = 0
for idx, dataset in enumerate(dataset_list):
    if files[idx] in skip_files:
        continue
    X, U = dataset['X'], dataset['Inp']

    if len(X) < num_rows_threshold:
        continue

    num_analysed_files += 1
    print(f'Loop {num_analysed_files}/176')

    # Determine the split index for the training and testing data
    split_index = int(np.floor(len(X) * 0.7))

    if split_index < trainlens[0]:
        continue
    
    #print('-' * 40)
    #print(f'{files[idx]}:')
    #print(f'Split index: {split_index}')

    # Split data
    X_train, X_test = X[:split_index], X[split_index:]
    U_train, U_test = U[:split_index], U[split_index:]

    #print(f'Valid ratio train {valid_train_ratio}')
    for trainlen in trainlens:
        if len(X_train) < trainlen:
            continue
        resulting_training_set = get_training_set(trainlen, X_train, U_train)
        if resulting_training_set == False:
            break
        X_train_trainlen, U_train_trainlen = resulting_training_set
        resulting_norms = norm_per_trainlen(X_train_trainlen, U_train_trainlen) 
        if resulting_norms == False:
            break
        frobenius_norm_A, frobenius_norm_K, l2_norm_AC = resulting_norms
        results_A[trainlen].append(frobenius_norm_A)
        results_K[trainlen].append(frobenius_norm_K)
        results_AC[trainlen].append(l2_norm_AC)
        print(f'Trainlen: {trainlen} real trainlen {len(X_train_trainlen)}')
    
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

# Compute mean and standard deviation for each ratio
mean_norms_A, sd_A, se_A = compute_me_sd_se(results_A)

mean_norms_K, sd_K, se_K = compute_me_sd_se(results_K)

mean_norms_AC, sd_AC, se_AC = compute_me_sd_se(results_AC)

folder = "results_trainlen"
df_A = pd.DataFrame({
    "trainlen": trainlens,
    "mean_norms": mean_norms_A,
    "sd": sd_A,
    "se": se_A
})
filepath = os.path.join(folder, filename_A)
df_A.to_csv(filepath, index=False)
print(f'Final results saved to {filename_A}')

df_K = pd.DataFrame({
    "trainlen": trainlens,
    "mean_norms": mean_norms_K,
    "sd": sd_K,
    "se": se_K
})
# Save to CSV
filepath = os.path.join(folder, filename_K)
df_K.to_csv(filepath, index=False)
print(f'Final results saved to {filename_K}')

df_AC = pd.DataFrame({
    "trainlen": trainlens,
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
for i,_ in enumerate(trainlens):
    print(f'Number of samples for valid ratio {trainlens[i]}: {num_elements[i]}')

# Reverse the data
trainlens_reversed = trainlens[::-1]
mean_norms_reversed = mean_norms_A[::-1]
std_errs_reversed = se_A[::-1]

# Create the plot
plt.figure(figsize=(8, 6))
plt.errorbar(trainlens_reversed, mean_norms_reversed, yerr=std_errs_reversed, fmt='-o', capsize=5, label='Frobenius A')

# Customize the plot
plt.title('Frobenius A vs Ratio')
plt.xlabel('Ratios')
plt.ylabel('Frobenius')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()

mean_norms_reversed = mean_norms_K[::-1]
std_errs_reversed = std_e_K[::-1]

# Create the plot
plt.figure(figsize=(8, 6))
plt.errorbar(trainlens_reversed, mean_norms_reversed, yerr=std_errs_reversed, fmt='-o', capsize=5, label='Frobenius K')

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
plt.errorbar(trainlens_reversed, mean_norms_reversed, yerr=std_errs_reversed, fmt='-o', capsize=5, label='Frobenius AC')

# Customize the plot
plt.title('Frobenius AC vs Ratio')
plt.xlabel('Ratios')
plt.ylabel('Frobenius')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()