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

# Function to compute the mean of the EMAS with missing data infront
def compute_mean_with_nan_infront(df, emas):
    df = df[emas]
    # Get the indices of the rows infront of with NaN
    indices_with_nan_infront = df[df.isna().any(axis=1)].index -1
    rows_with_nan_infront = df.loc[indices_with_nan_infront]
    column_means_nan = rows_with_nan_infront.mean(skipna=True)
    return column_means_nan

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
subfolders = ["MRT1","MRT2","MRT3"]

# Collect all CSV files with less than 50% missing rows from the specified subfolders
csv_files = []
for subfolder in subfolders:
    folder_path = os.path.join(prep_data_folder, subfolder, "*.csv")
    for file in glob.glob(folder_path):
        df = pd.read_csv(file)
        missing_data_percentage = compute_missing_data_percentage(df)
        if missing_data_percentage < 0.8:
            csv_files.append(file)

# Initialize lists to store results
column_means_list = []
column_means_nan_list = []
deterioration_list = []
missing_data_list = []

# Process each CSV file
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    column_means = compute_mean_without_nan(df, emas)
    column_means_nan = compute_mean_with_nan_infront(df, emas)
    deterioration = compute_deterioration_means(column_means, column_means_nan)
    missing_data = compute_missing_data_percentage(df)
    
    column_means_list.append(column_means)
    column_means_nan_list.append(column_means_nan)
    deterioration_list.append(deterioration)
    missing_data_list.append(missing_data)

# Compute the mean across all datasets
column_means = sum(column_means_list) / len(column_means_list)
column_means_nan = sum(column_means_nan_list) / len(column_means_nan_list)
deteriorations = sum(deterioration_list) / len(deterioration_list)
missing_data_mean = sum(missing_data_list) / len(missing_data_list)

"""
# Perform the Shapiro-Wilk Test for normality
check_aprox_normal = np.array(deteriorations)
statistic, p_value = shapiro(check_aprox_normal)
print(f"Shapiro-Wilk Test Statistic: {statistic}")
print(f"p-value: {p_value}")
"""

# Compute z-scores for deterioration
z_scores = compute_z_score_deterioration(deteriorations)

# Concatenate results into a DataFrame
concatenated = pd.concat([column_means, column_means_nan, deteriorations, z_scores], axis=1)
concatenated.columns = ['means', 'means before missing data', 'deterioration', 'z_score']
print(concatenated)

# Compute the mean deterioration over all datasets
mean_deterioration = deteriorations.sum() / len(emas)
print(f'Deterioration mean: {mean_deterioration}')

# Findings:
# Higher tiredness, stress are indicator for missing data
# Indicates that the missing data is not random, but more data needs to be analyzed to confirm this
print()
####################################################################################################################################

# Print missing data percentages
for i, missing_data in enumerate(missing_data_list, start=1):
    print(f'Missing data percentage {i}: {missing_data}')

print(f'Missing data percentage mean: {missing_data_mean}')