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
length_reduction = 20 # When this is set to None the Eigenvalues are computed for each trainlen

# Write results to this file
filename = "trainlen-eigenvaluesA_len160.csv"

#files_ratio80 = ['data\\MRT1/processed_csv_no_con\\11228_15.csv', 'data\\MRT1/processed_csv_no_con\\11228_19.csv', 'data\\MRT1/processed_csv_no_con\\11228_28.csv', 'data\\MRT1/processed_csv_no_con\\11228_34.csv', 'data\\MRT1/processed_csv_no_con\\11228_35.csv', 'data\\MRT1/processed_csv_no_con\\11228_52.csv', 'data\\MRT2/processed_csv_no_con\\12600_30.csv', 'data\\MRT2/processed_csv_no_con\\12600_32.csv', 'data\\MRT2/processed_csv_no_con\\12600_64.csv', 'data\\MRT2/processed_csv_no_con\\12600_65.csv', 'data\\MRT3/processed_csv_no_con\\12600_218.csv', 'data\\MRT3/processed_csv_no_con\\12600_221.csv', 'data\\MRT3/processed_csv_no_con\\12600_228.csv', 'data\\MRT3/processed_csv_no_con\\12600_239.csv', 'data\\MRT3/processed_csv_no_con\\12600_241.csv', 'data\\MRT3/processed_csv_no_con\\12600_261.csv']

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

def get_valid_ratio(data):
    # Valid rows are rows without nan value where also the next row has no nan value
    valid = ~np.isnan(data).any(axis=1)
    valid_rows = valid[:-1] & valid[1:]
    total = valid_rows.sum()
    return total/len(data)

def compute_eigenvalues(X_train_ratio, U_train_ratio):
    X_train_ratio = locf(X_train_ratio)
    A, B, lmbda = utils.stable_ridge_regression(X_train_ratio, U_train_ratio)
    eigenvalues = np.linalg.eigvals(A)
    return eigenvalues

# Dictionary to store ratios and corresponding frobenius norms
results_trainlens = []
results_eigenvalues = []

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

    if split_index < trainlens[0]:
        continue

    num_analysed_files += 1
    print(f'Loop {num_analysed_files}/176')
    #print('-' * 40)
    #print(f'{files[idx]}:')
    #print(f'Split index: {split_index}')
    
    # Split data
    X_train, X_test = X[:split_index], X[split_index:]
    U_train, U_test = U[:split_index], U[split_index:]

    if length_reduction != None:
        initial_trainlen = len(X_train)
        skip = False
        eigenvalues = compute_eigenvalues(X_train, U_train)
        dominant_eigenvalue = np.max(np.abs(eigenvalues))
        results_trainlens.append(initial_trainlen)
        results_eigenvalues.append(dominant_eigenvalue)
        #print(f'Initial Trainlen: {initial_trainlen}; real_trainlen: {len(X_train)}; dominant eigenvalue: {dominant_eigenvalue}')
        #print(f'Valid ratio train {valid_train_ratio}')
        iterations = 1
        target_trainlen = initial_trainlen - length_reduction
        while target_trainlen > 10:
            # Do multiple ratioed dataset creations -> multiple different removel processes -> decrease standard deviation caused by individual random removal process
            target_trainlen = initial_trainlen - length_reduction * iterations
            if target_trainlen < 1:
                break
            X_train_trainlen, U_train_trainlen = get_training_set(target_trainlen, X_train, U_train)
            valid_ratio = get_valid_ratio(X_train_trainlen)
            if valid_ratio == 0.0:
                break
            eigenvalues = compute_eigenvalues(X_train_trainlen, U_train_trainlen)
            dominant_eigenvalue = np.max(np.abs(eigenvalues))
            results_trainlens.append(target_trainlen)
            results_eigenvalues.append(dominant_eigenvalue)
            #print(f'Trainlen: {target_trainlen}; real_trainlen: {len(X_train_trainlen)}; dominant eigenvalue: {dominant_eigenvalue}')
            iterations += 1
    else:
        for trainlen in trainlens:
            if len(X_train) < trainlen:
                continue
            X_train_trainlen, U_train_trainlen = get_training_set(trainlen, X_train, U_train)
            valid_ratio = get_valid_ratio(X_train_trainlen)
            if valid_ratio == 0.0:
                continue
            eigenvalues = compute_eigenvalues(X_train_trainlen, U_train_trainlen)
            dominant_eigenvalue = np.max(np.abs(eigenvalues))
            if dominant_eigenvalue == 0:
                print("eigenvalue 0")
                print(files[idx])
            results_trainlens.append(trainlen)
            results_eigenvalues.append(dominant_eigenvalue)
            #print(f'Trainlen: {trainlen}; real_trainlen: {len(X_train_trainlen)}; dominant eigenvalue: {dominant_eigenvalue}')


from collections import Counter
element_counts = Counter(results_trainlens)
for element, count in element_counts.items():
    print(f"{element}: {count} times")
        
# Convert results to a DataFrame to save them to a csv file
df_final = pd.DataFrame({
    "trainlen": results_trainlens,
    "eigenvalues": results_eigenvalues,
})

# Save to CSV
filepath = os.path.join("results_trainlen", filename)
df_final.to_csv(filepath, index=False)
print(f'Final results saved to {filename}')

print(f'Datasets analysed {num_analysed_files}')
print(f'Number of datapoints: {len(results_trainlens)}')

# Reverse the data
trainlens_reversed = results_trainlens[::-1]
dominant_eigenvalues_reversed = results_eigenvalues[::-1]

# Create scatter plot: valid data ratio vs. dominant eigenvalue magnitude
plt.figure(figsize=(8, 6))
plt.scatter(trainlens_reversed, dominant_eigenvalues_reversed, color='b', label='Dominant Eigenvalue')
plt.xlabel('Valid Data Ratio')
plt.ylabel('Dominant Eigenvalue Magnitude')
plt.title('Dominant Eigenvalue vs. Valid Data Ratio')
plt.grid(True)
plt.legend()
plt.show()