import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ctrl import utils
import numpy as np
import json

### Descriptive statistics of the dataset (5.1.1, Table 2)

emas = ['EMA_mood', 'EMA_disappointed', 'EMA_scared', 'EMA_worry', 'EMA_down', 'EMA_sad', 'EMA_confidence', 'EMA_stress', 'EMA_lonely', 'EMA_energetic', 'EMA_concentration', 'EMA_resilience', 'EMA_tired', 'EMA_satisfied', 'EMA_relaxed']
emis = ['interactive1', 'interactive2', 'interactive3', 'interactive4', 'interactive5', 'interactive6', 'interactive7', 'interactive8']

data_folder = "data"
subfolders = ["MRT1/processed_csv_no_con","MRT2/processed_csv_no_con","MRT3/processed_csv_no_con"]
dataset_list, files = utils.load_dataset(data_folder, subfolders, emas, emis, centered=False)

def get_valid_ratio(data):
    # Valid rows are rows without nan value where also the next row has no nan value
    valid = ~np.isnan(data).any(axis=1)
    valid_rows = valid[:-1] & valid[1:]
    total = valid_rows.sum()
    return total/len(data)

num_rows_list = []
means_list = []
variation_emas_list = []
missing_ratio_list = []
row_differences_list = []
valid_ratio_list = []
num_interventions_list = []
interventions_ratio_list = []
num_ones = 0
num_twos = 0
num_threes = 0
num_fours = 0

num_analysed_files = 0

# Compute descriptive statistics for the dataset (excluding files that arent present in the final evaluation of the control strategies)
with open('exclude_participant_list.json', 'r') as f:
    exclude_participant_list = json.load(f)

for dataset, file in zip(dataset_list, files):
    if file in exclude_participant_list:
        # Participants for whom there is no trained RNN model are excluded from all analyses
        continue
    num_analysed_files += 1
    X, U = dataset['X'], dataset['Inp']

    if np.isnan(X[0,:]).any():
        print("First row contains NaN values")

    num_rows_list.append(len(X))

    mean_emas = np.nanmean(X)
    means_list.append(mean_emas)

    variation_emas = np.mean(np.nanstd(X, axis=0))
    variation_emas_list.append(variation_emas)

    row_differences =  np.abs(np.diff(X, axis=0))
    mean_row_diff = np.nanmean(row_differences, axis=0)
    row_differences_list.append(np.mean(mean_row_diff))

    missing_rows = np.isnan(X).all(axis=1)
    missing_row_ratio = np.sum(missing_rows) / len(X)
    missing_ratio_list.append(missing_row_ratio)

    valid_ratio = get_valid_ratio(X)
    valid_ratio_list.append(valid_ratio)

    num_intervention = np.sum(U)
    num_interventions_list.append(num_intervention)

    intervention_ratio = len(X) / num_intervention
    interventions_ratio_list.append(intervention_ratio)

    num_ones += np.sum(U == 1)
    num_twos += np.sum(U == 2)
    num_threes += np.sum(U == 3)
    num_fours += np.sum(U == 4)
    
num_rows_array = np.array(num_rows_list)
means_array = np.array(means_list)
variation_emas_array = np.array(variation_emas_list)
missing_ratio_array = np.array(missing_ratio_list)
row_differences_array = np.array(row_differences_list)
valid_ratio_array = np.array(valid_ratio_list)
num_interventions_array = np.array(num_interventions_list)
interventions_ratio_array = np.array(interventions_ratio_list)

arrays = {
'num_rows': num_rows_array,
'means': means_array,
'variation_emas': variation_emas_array,
'missing_ratio': missing_ratio_array,
'row_differences': row_differences_array,
'valid_ratio': valid_ratio_array,
'num_interventions': num_interventions_array,
'interventions_ratio': interventions_ratio_array
}
print(f'num analysed files: {num_analysed_files}')
for name, arr in arrays.items():
    print(f"{name}: {np.mean(arr):.4f}, ({np.std(arr):.4f})")

#print(f'1 {num_ones}, 2 {num_twos}, 3 {num_threes}, 4 {num_fours}')

print(f'overall mean: {sum(means_list)/ len(means_list)}')

num_interventions_list_N = []
num_valid_rows_list = []
for dataset, file in zip(dataset_list, files):
    if file in exclude_participant_list:
        # Participants for whom there is no trained RNN model are excluded from all analyses
        continue
    
    X, U = dataset['X'], dataset['Inp']
    num_intervention = np.sum(U)
    num_interventions_list_N.append(num_intervention)

    valid = ~np.isnan(X).any(axis=1)
    valid_rows = valid[:-1] & valid[1:] # valid rows are those where the predictor and target are both valid (no NaN values)
    total = valid_rows.sum() # total number of valid rows
    num_valid_rows_list.append(total)

print()
means_list.sort()
N = 50  # Number of lowest mean files to consider (also considering missing models)
u = means_list[N - 1]
print(f"{N}-th lowest ema mean: {u}")

num_interventions_list_N.sort(reverse=True)
num_intervention_threshold = num_interventions_list_N[N - 1]
print(f"{N}-th highest num intervention: {num_intervention_threshold}")

num_valid_rows_list.sort(reverse=True)
valid_rows_threshold = num_valid_rows_list[N - 1]
print(f"{N}-th highest num valid rows: {valid_rows_threshold}")
