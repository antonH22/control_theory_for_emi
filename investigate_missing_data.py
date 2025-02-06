import pandas as pd
import numpy as np
from scipy.stats import shapiro
import os
import glob

emas = ['EMA_mood', 'EMA_disappointed', 'EMA_scared', 'EMA_worry', 'EMA_down', 'EMA_sad', 'EMA_confidence', 'EMA_stress', 'EMA_lonely', 'EMA_energetic', 'EMA_concentration', 'EMA_resilience', 'EMA_tired', 'EMA_satisfied', 'EMA_relaxed']

# Function to compute the mean of the EMAS without missing data
def compute_mean_without_nan(df, emas):
    df = df[emas]
    rows_without_nan = df.dropna()
    column_means = rows_without_nan.mean(skipna=True)
    return column_means

# Function to compute the mean of the EMAS before/after missing data
def compute_mean_around_nan(df, emas):
    df = df[emas]
    missing_rows = df.isna().any(axis=1)
    # Get indices immediately BEFORE missing data
    start_of_missing = missing_rows & (~missing_rows.shift(1, fill_value=False))
    indices_before_missing = start_of_missing[start_of_missing].index - 1

    # Get indices immediately AFTER missing data
    end_of_missing = (~missing_rows) & (missing_rows.shift(1, fill_value=False))
    indices_after_missing = end_of_missing[end_of_missing].index

    rows_before_nan = df.loc[indices_before_missing]
    column_means_before_nan = rows_before_nan.mean(skipna=True)

    rows_after_nan = df.loc[indices_after_missing]
    column_means_after_nan = rows_after_nan.mean(skipna=True)

    return column_means_before_nan, column_means_after_nan

#csv1 = "prep_data/MRT1/11228_12_prep.csv"
#df1 = pd.read_csv(csv1) 
#cm1 = compute_mean_around_nan(df1, emas)


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
    df = df[emas]
    num_rows_with_missing_data = df.isna().any(axis=1).sum()  # Count rows with missing values
    return num_rows_with_missing_data / len(df)


# Get all CSV files in the preprocessed data folder
prep_data_folder = "prep_data"
subfolders = ["MRT1", "MRT2", "MRT3"]

# Collect all CSV files with less than 50% missing rows from the specified subfolders
csv_files = []
for subfolder in subfolders:
    folder_path = os.path.join(prep_data_folder, subfolder, "*.csv")
    for file in glob.glob(folder_path):
        df = pd.read_csv(file)
        missing_data_percentage = compute_missing_data_percentage(df)
        if missing_data_percentage < 0.5:
            csv_files.append(file)

# Initialize lists to store results
column_means_list = []
column_means_before_nan_list = []
column_means_after_nan_list = []
deterioration_before_list = []
deterioration_after_list = []
missing_data_list = []

# Process each CSV file
for i,csv_file in enumerate(csv_files):
    df = pd.read_csv(csv_file)
    column_means = compute_mean_without_nan(df, emas)
    column_means_before_nan, column_means_after_nan = compute_mean_around_nan(df, emas)
    deterioration_before = compute_deterioration_means(column_means, column_means_before_nan)
    deterioration_after = compute_deterioration_means(column_means, column_means_after_nan)
    missing_data = compute_missing_data_percentage(df)
    
    column_means_list.append(column_means)
    column_means_before_nan_list.append(column_means_before_nan)
    column_means_after_nan_list.append(column_means_after_nan)
    deterioration_before_list.append(deterioration_before)
    deterioration_after_list.append(deterioration_after)
    missing_data_list.append(missing_data)

# Compute the mean across all datasets
column_means = sum(column_means_list) / len(column_means_list)
column_means_before_nan = sum(column_means_before_nan_list) / len(column_means_before_nan_list)
column_means_after_nan = sum(column_means_after_nan_list) / len(column_means_after_nan_list)
deteriorations_before = sum(deterioration_before_list) / len(deterioration_before_list)
deteriorations_after = sum(deterioration_after_list) / len(deterioration_after_list)
missing_data_mean = sum(missing_data_list) / len(missing_data_list)


# Perform the Shapiro-Wilk Test for normality
#check_aprox_normal = np.array(deteriorations)
#statistic, p_value = shapiro(check_aprox_normal)
#print(f"Shapiro-Wilk Test Statistic: {statistic}")
#print(f"p-value: {p_value}")


# Compute z-scores for deterioration
z_scores_before = compute_z_score_deterioration(deteriorations_before)
z_scores_after = compute_z_score_deterioration(deteriorations_after)

print(f'{len(csv_files)} datasets were analyzed')
# Concatenate results into a DataFrame
concatenated = pd.concat([column_means, column_means_before_nan, deteriorations_before, z_scores_before, column_means_after_nan, z_scores_after], axis=1)
concatenated.columns = ['means', 'means before nan', 'deterioration before nan', 'z_score', 'means after nan', 'z_score (after nan)']
print(concatenated)

# Compute the mean deterioration over all datasets
mean_deterioration = deteriorations_before.sum() / len(emas)
print(f'Deterioration (before) mean: {mean_deterioration}')

mean_deterioration_after = deteriorations_after.sum() / len(emas)
print(f'Deterioration (after) mean: {mean_deterioration_after}')

# Findings:
# Higher tiredness, stress are indicator for missing data
# Indicates that the missing data is not random, but more data needs to be analyzed to confirm this
print()
####################################################################################################################################
# Print missing data percentages
print(f'File : Missing data')
for i, missing_data in enumerate(missing_data_list):
    print(f'{csv_files[i]}: {missing_data:.2%}')

print(f'Missing data percentage mean: {missing_data_mean:.2%}')