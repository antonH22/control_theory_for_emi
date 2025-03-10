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
trainlens = [120,110,100,90,80,70,60,50,40,30,20] # When split index = 0.7
#trainlens = [200,175,150,125,100,75,50,25] # When setting split index = 1.0

# Set the filenames of the results
filename_A = f'trainlen-frobeniusA_07.csv'
filename_K = f'trainlen-frobeniusK_07.csv'
filename_AC = f'trainlen-frobeniusAC_07.csv'

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

def norm_per_trainlen(X_train_trainlen, U_train_trainlen):
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
        #print(f'Trainlen: {trainlen} real trainlen {len(X_train_trainlen)}')
    
# Compute mean and standard deviation for each ratio
mean_norms_A = [np.mean(norms) for norms in results_A.values()]
std_devs_A = [np.std(norms) for norms in results_A.values()]

mean_norms_K = [np.mean(norms) for norms in results_K.values()]
std_devs_K = [np.std(norms) for norms in results_K.values()]

mean_norms_AC = [np.mean(norms) for norms in results_AC.values()]
std_devs_AC = [np.std(norms) for norms in results_AC.values()]

# Convert results to a DataFrame to save them to a csv file
folder = "results_trainlen"
df_A = pd.DataFrame({
    "trainlen": trainlens,
    "mean_norms": mean_norms_A,
    "std_dev": std_devs_A
})
filepath = os.path.join(folder, filename_A)
df_A.to_csv(filepath, index=False)
print(f'Final results saved to {filename_A}')

df_K = pd.DataFrame({
    "trainlen": trainlens,
    "mean_norms": mean_norms_K,
    "std_dev": std_devs_K
})
# Save to CSV
filepath = os.path.join(folder, filename_K)
df_K.to_csv(filepath, index=False)
print(f'Final results saved to {filename_K}')

df_AC = pd.DataFrame({
    "trainlen": trainlens,
    "mean_norms": mean_norms_AC,
    "std_dev": std_devs_AC
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
std_devs_reversed = std_devs_A[::-1]

# Create the plot
plt.figure(figsize=(8, 6))
plt.errorbar(trainlens_reversed, mean_norms_reversed, yerr=std_devs_reversed, fmt='-o', capsize=5, label='Frobenius A')

# Customize the plot
plt.title('Frobenius A vs Ratio')
plt.xlabel('Ratios')
plt.ylabel('Frobenius')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()

mean_norms_reversed = mean_norms_K[::-1]
std_devs_reversed = std_devs_K[::-1]

# Create the plot
plt.figure(figsize=(8, 6))
plt.errorbar(trainlens_reversed, mean_norms_reversed, yerr=std_devs_reversed, fmt='-o', capsize=5, label='Frobenius K')

# Customize the plot
plt.title('Frobenius K vs Ratio')
plt.xlabel('Ratios')
plt.ylabel('Frobenius')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()

mean_norms_reversed = mean_norms_AC[::-1]
std_devs_reversed = std_devs_AC[::-1]

# Create the plot
plt.figure(figsize=(8, 6))
plt.errorbar(trainlens_reversed, mean_norms_reversed, yerr=std_devs_reversed, fmt='-o', capsize=5, label='Frobenius AC')

# Customize the plot
plt.title('Frobenius AC vs Ratio')
plt.xlabel('Ratios')
plt.ylabel('Frobenius')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()