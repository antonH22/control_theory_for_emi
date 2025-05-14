import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))

import pandas as pd
import numpy as np
import os
import glob
import json

from collections import Counter

### Investigating missing data patterns (5.1.2, Table 3)

emas = ['EMA_mood', 'EMA_disappointed', 'EMA_scared', 'EMA_worry', 'EMA_down', 'EMA_sad', 'EMA_confidence', 'EMA_stress', 'EMA_lonely', 'EMA_energetic', 'EMA_concentration', 'EMA_resilience', 'EMA_tired', 'EMA_satisfied', 'EMA_relaxed']
emis = ['interactive1', 'interactive2', 'interactive3', 'interactive4', 'interactive5', 'interactive6', 'interactive7', 'interactive8']

# Function to compute the mean of the EMAS without missing data
def compute_mean_without_nan(df):
    rows_without_nan = df.dropna()
    column_means = rows_without_nan.mean(skipna=True)
    return column_means

# Used to compute the mean over all csvs
def select_rows_before_nan(df):
    # Get indices immediately BEFORE missing data
    missing_rows = df.isna().any(axis=1)
    start_of_missing = missing_rows & (~missing_rows.shift(1, fill_value=False))
    indices_before_missing = start_of_missing[start_of_missing].index - 1
    rows_before_nan = df.loc[indices_before_missing]
    return rows_before_nan

def select_rows_after_nan(df):
    # Get indices immediately AFTER missing data
    missing_rows = df.isna().any(axis=1)
    end_of_missing = missing_rows & (~missing_rows.shift(-1, fill_value=False))
    indices_after_missing = end_of_missing[end_of_missing].index + 1
    indices_after_missing = indices_after_missing[indices_after_missing < len(df)]
    rows_after_nan = df.loc[indices_after_missing]
    return rows_after_nan

def select_random_rows(df):
    missing_rows = df.isna().any(axis=1)
    start_of_missing = missing_rows & (~missing_rows.shift(1, fill_value=False))
    indices_before_missing = start_of_missing[start_of_missing].index
    num_rows = len(indices_before_missing)
    valid_rows = df.dropna()
    sampled_rows = valid_rows.sample(n=num_rows)
    return sampled_rows

# Function to compute the mean of the EMAS before/after missing data
def compute_mean_around_nan(df):
    rows_before_nan = select_rows_before_nan(df)
    column_means_before_nan = rows_before_nan.mean(skipna=True)
    rows_after_nan = select_rows_after_nan(df)
    column_means_after_nan = rows_after_nan.mean(skipna=True)
    return column_means_before_nan, column_means_after_nan

# Function to compute the deterioration of the means before the nan and the general means
def compute_deterioration_means(column_means, column_means_nan):
    deterioration = column_means_nan - column_means
    return deterioration

def compute_z_score_deterioration(deterioration):
    mean_deterioration = deterioration.sum() / len(emas)
    std_deterioration = deterioration.std()
    z_score = (deterioration - mean_deterioration) / std_deterioration
    return z_score

def compute_missing_data_percentage(df):
    num_rows_with_missing_data = df.isna().any(axis=1).sum()  # Count rows with missing values
    return num_rows_with_missing_data / len(df)

# Function to compute the mean of random data to compare deterioration and z_score
def compute_mean_random_rows(df):
    sampled_rows = select_random_rows(df)
    mean_values = sampled_rows.mean()
    return mean_values

def bootstrap_ci(data, func=np.mean, n_bootstrap=10000, ci=95, seed=None):
    " Computes a bootstrap confidence interval for a given statistic (here: mean deviation of observations preceding missing data). "
    if seed is not None:
        np.random.seed(seed)
    
    bootstrap_samples = np.array([func(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)])
    lower, upper = np.percentile(bootstrap_samples, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return lower, upper

def apply_bootstrapping(df, n_bootstrap=10000, ci=95, seed=None):
    " Applies bootstrapping to each column in the dataframe that holds the deteriorations for each EMA variable "
    bootstrap_results = {}

    for ema in df.columns:
        data = df[ema].values
        lower, upper = bootstrap_ci(data, n_bootstrap=n_bootstrap, ci=ci, seed=seed)
        bootstrap_results[ema] = [lower, upper]

    # Convert the results to a DataFrame for better readability
    df_bootstrap = pd.DataFrame.from_dict(bootstrap_results, orient='index', columns=['CI_Lower', 'CI_Upper'])

    return df_bootstrap

def print_overall_results(column_means, column_means_before_nan, deteriorations_before_nan, deterioration_random, overall_deteriorations_df):
    " Computes and prints the results for averaged deteriorations across all participants "
    z_scores = compute_z_score_deterioration(deteriorations_before_nan)
    z_scores_random = compute_z_score_deterioration(deterioration_random)

    df_bootstrap = apply_bootstrapping(overall_deteriorations_df, n_bootstrap=100000, seed=1)
    # Bootstrap confidence intervals for all deterioration values per ema (if the confidence interval does not include 0 the deterioration is significantly different from 0)
    significant_bootstrap = (df_bootstrap["CI_Lower"] > 0) | (df_bootstrap["CI_Upper"] < 0)

    concatenated = pd.concat([column_means, column_means_before_nan, deteriorations_before_nan, significant_bootstrap, z_scores, deterioration_random, z_scores_random], axis=1)
    concatenated.columns = ['means', 'means nan', 'deterioration','significant (bootstr.)', 'z-score','deteriorations random', 'z-score (random)']
    print(concatenated)
    print()
    print("Bootstrapping confidence intervals for deterioration values:")
    # Join the bootstrap results with the original deterioration values
    df_deterioration = pd.DataFrame({"Deterioration": deteriorations_before_nan})
    # Display the final table
    df_bootstrap_results = df_deterioration.join(df_bootstrap)
    print(df_bootstrap_results)
    print('-'*100)
    
def number_rows_before_nan(file):
    # Get indices immediately BEFORE missing data
    df = pd.read_csv(file)
    df = df[emas]
    missing_rows = df.isna().any(axis=1)
    start_of_missing = missing_rows & (~missing_rows.shift(1, fill_value=False))
    indices_before_missing = start_of_missing[start_of_missing].index - 1
    return len(indices_before_missing)

#############################################################################################################

data_folder = "data"
subfolders = ["MRT1/processed_csv_no_con","MRT2/processed_csv_no_con","MRT3/processed_csv_no_con"]
csv_files = []
for subfolder in subfolders:
    folder_path = os.path.join(data_folder, subfolder, "*.csv")
    for file in glob.glob(folder_path):
        csv_files.append(file)

# Initialize lists to store results
column_means_list = []
column_means_before_nan_list = []
column_means_after_nan_list = []
column_means_random_list = []
deterioration_before_list = []
deterioration_after_list = []
deterioration_random_list = []
z_score_before_list = []
missing_data_list = []
z_score_random_list = []

rows_before_nan_overall_df = pd.DataFrame()
random_rows = pd.DataFrame() # The number of random rows is equal to the number of rows before nan
overall_deteriorations_df = pd.DataFrame()
overall_random_deteriorations_df = pd.DataFrame()
overall_num_significant_bootstrap = 0
overall_num_significant_bootstrap_random = 0

overall_significant_emas_deterioration_list = []
overall_significant_emas_improvement_list = []
overall_significant_emas_deterioration_random_list = []
overall_significant_emas_improvement_random_list = []

overall_num_alone_ratio = []
overall_num_alone_before_nan_ratio = []

num_analysed_files = 0

with open('exclude_participant_list.json', 'r') as f:
    exclude_participant_list = json.load(f)
# Process each CSV file: Compute and print results for each participant
number_participants_without_significant_deviation = 0
for csv_file  in csv_files:
    if csv_file in exclude_participant_list:
        # Participants for whom there is no trained RNN model are excluded from all analyses
        continue
    num_analysed_files += 1
    rows_before_nan_df = pd.DataFrame()
    csv_df = pd.read_csv(csv_file)
    # Delete empty rows in the beginning
    df = csv_df[emas]
    first_non_na_index = df.notna().all(axis=1).idxmax()
    df = df.iloc[first_non_na_index:].reset_index(drop=True)

    column_means = compute_mean_without_nan(df)
    column_means_before_nan, column_means_after_nan = compute_mean_around_nan(df)
    column_means_random = compute_mean_random_rows(df)
    deterioration_before = compute_deterioration_means(column_means, column_means_before_nan)
    deterioration_after = compute_deterioration_means(column_means, column_means_after_nan)
    deterioration_random = compute_deterioration_means(column_means, column_means_random)
    missing_data = compute_missing_data_percentage(df)
    z_scores_before = compute_z_score_deterioration(deterioration_before)
    z_scores_random = compute_z_score_deterioration(deterioration_random)

    rows_before_nan_df = select_rows_before_nan(df)
    rows_before_nan_overall_df = pd.concat([rows_before_nan_overall_df, rows_before_nan_df], axis = 0)

    random_rows_df = pd.concat([random_rows, select_random_rows(df)], axis = 0)

    random_deteriorations_df = random_rows_df - column_means
    overall_deteriorations_df = pd.concat([overall_random_deteriorations_df, random_deteriorations_df], axis = 0)

    deteriorations_df = rows_before_nan_df - column_means
    overall_deteriorations_df = pd.concat([overall_deteriorations_df, deteriorations_df], axis = 0)

    # Bootstrap confidence intervals for deterioration values per ema (if the confidence interval does not include 0 the deterioration is significantly different from 0)

    bootstrap_df = apply_bootstrapping(deteriorations_df, seed=1)

    significant_bootstrap_deterioration = (bootstrap_df["CI_Lower"] > 0)
    significant_emas_deterioration = bootstrap_df.index[significant_bootstrap_deterioration].tolist()
    overall_significant_emas_deterioration_list.extend(significant_emas_deterioration)
    num_significant_bootstrap_participant = significant_bootstrap_deterioration.sum()

    significant_bootstrap_improvement = (bootstrap_df["CI_Upper"] < 0)
    significant_emas_improvement = bootstrap_df.index[significant_bootstrap_improvement].tolist()
    overall_significant_emas_improvement_list.extend(significant_emas_improvement)
    num_significant_bootstrap_participant += significant_bootstrap_improvement.sum()

    # Bootstrap confidence intervals for random deterioration values
    bootstrap_random_df = apply_bootstrapping(random_deteriorations_df, seed=1)
    significant_bootstrap_deterioration_random = (bootstrap_random_df["CI_Lower"] > 0)
    significant_emas_deterioration_random = bootstrap_random_df.index[significant_bootstrap_deterioration_random].tolist()
    overall_significant_emas_deterioration_random_list.extend(significant_emas_deterioration_random)

    significant_bootstrap_improvement_random = (bootstrap_random_df["CI_Upper"] < 0)
    significant_emas_improvement_random = bootstrap_random_df.index[significant_bootstrap_improvement_random].tolist()
    overall_significant_emas_improvement_random_list.extend(significant_emas_improvement_random)

    # Print results for each dataset
    #print_results(csv_file, column_means, column_means_before_nan, deterioration_before, deterioration_after, deterioration_random, bootstrap_df, bootstrap_random_df)
    print(f'{csv_file}:')
    print(f'Emas with significant deterioration: {significant_emas_deterioration}')
    print(f'Emas with significant improvement: {significant_emas_improvement}')
    print(f'Emas with significant deterioration random: {significant_emas_deterioration_random}')
    print(f'Emas with significant improvement random: {significant_emas_improvement_random}')
    print('-'*100)
    # Convert to DataFrame

    if num_significant_bootstrap_participant == 0:
        number_participants_without_significant_deviation += 1

    column_means_list.append(column_means)
    column_means_before_nan_list.append(column_means_before_nan)
    column_means_after_nan_list.append(column_means_after_nan)
    column_means_random_list.append(column_means_random)
    deterioration_before_list.append(deterioration_before)
    deterioration_after_list.append(deterioration_after)
    deterioration_random_list.append(deterioration_random)
    missing_data_list.append(missing_data)
    z_score_before_list.append(z_scores_before)
    z_score_random_list.append(z_scores_random)


# Compute the mean across all participants
column_means = sum(column_means_list) / len(column_means_list)
column_means_before_nan = sum(column_means_before_nan_list) / len(column_means_before_nan_list)
column_means_before_nan = rows_before_nan_overall_df.mean()
deteriorations_before_nan= compute_deterioration_means(column_means, column_means_before_nan)
column_means_after_nan = sum(column_means_after_nan_list) / len(column_means_after_nan_list)
column_means_random = sum(column_means_random_list) / len(column_means_random_list)
deteriorations_before = sum(deterioration_before_list) / len(deterioration_before_list)
deteriorations_after = sum(deterioration_after_list) / len(deterioration_after_list)
deterioration_random = compute_deterioration_means(column_means, random_rows_df.mean())
missing_data_mean = sum(missing_data_list) / len(missing_data_list)

print('#'*100)
print(f'Number of datasets analyzed: {num_analysed_files}')
print(f'Number of participants that had no significant deviation: {number_participants_without_significant_deviation}')
print_overall_results(column_means, column_means_before_nan, deteriorations_before_nan, deterioration_random, overall_deteriorations_df)

print(f'Number of significant deterioration before nan (bootstrap overall): {overall_num_significant_bootstrap}')
print(f'Compared to number of random missingness (bootstrap random overall): {overall_num_significant_bootstrap_random}')

ema_counts_deterioration = Counter(overall_significant_emas_deterioration_list)
ema_counts_improvement = Counter(overall_significant_emas_improvement_list)
ema_counts_deterioration_random = Counter(overall_significant_emas_deterioration_random_list)
ema_counts_improvement_random = Counter(overall_significant_emas_improvement_random_list)

df_significant_counts = pd.DataFrame.from_dict(
    {"Deteriorations": ema_counts_deterioration, "Deteriorations random": ema_counts_deterioration_random, "Improvements": ema_counts_improvement, "Improvements random": ema_counts_improvement_random},
).fillna(0)
df_counts_deteriorations_sorted = df_significant_counts.sort_values(by='Deteriorations', ascending=False)
print(df_counts_deteriorations_sorted)

# Findings: 
# - No general indicator (over all participants) for missing data was found, but individual indicators can be found for some participants.



# Bootstrapping is the better approach compared to z-score here, because you dont have to assume that the EMA data is normally distributed
"""
count_significant_deterioration = sum((np.abs(z_score) > 2).sum() for z_score in z_score_before_list)
count_significant_deterioration_random = sum((np.abs(z_score) > 2).sum() for z_score in z_score_random_list)
print(f'Number of significant deterioration before nan (z-score > 2): {count_significant_deterioration}')
print(f'Compared to z-score of random missingness: {count_significant_deterioration_random}')
"""

