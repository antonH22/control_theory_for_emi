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

# Function to compute the mean of random data to compare deterioration and z_score
def compute_mean_random_rows(df, emas):
    missing_rows = df.isna().any(axis=1)
    start_of_missing = missing_rows & (~missing_rows.shift(1, fill_value=False))
    indices_before_missing = start_of_missing[start_of_missing].index
    num_rows = len(indices_before_missing)
    valid_rows = df[emas].dropna()
    sampled_rows = valid_rows.sample(n=num_rows)
    mean_values = sampled_rows.mean()
    return mean_values

def print_results(csv_file, column_means, column_means_before_nan, deteriorations_before, deteriorations_after, deterioration_random):
    # Concatenate results into a DataFrame
    z_scores_before = compute_z_score_deterioration(deteriorations_before)
    z_scores_after = compute_z_score_deterioration(deteriorations_after)
    z_scores_random = compute_z_score_deterioration(deterioration_random)
    print(f'{csv_file}:')
    concatenated = pd.concat([column_means, column_means_before_nan, deteriorations_before, z_scores_before, z_scores_after, z_scores_random], axis=1)
    concatenated.columns = ['means', 'means before nan', 'deteriorations before nan', 'z_scores', 'z_scores (after nan)', 'z_scores (random)']
    print(concatenated)
    print()

# This block will only run if the script is executed directly
if __name__ == "__main__":
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
            if missing_data_percentage < 1:
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

    # Process each CSV file
    for i,csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        column_means = compute_mean_without_nan(df, emas)
        column_means_before_nan, column_means_after_nan = compute_mean_around_nan(df, emas)
        column_means_random = compute_mean_random_rows(df, emas)
        deterioration_before = compute_deterioration_means(column_means, column_means_before_nan)
        deterioration_after = compute_deterioration_means(column_means, column_means_after_nan)
        deterioration_random = compute_deterioration_means(column_means, column_means_random)
        missing_data = compute_missing_data_percentage(df)
        z_scores_before = compute_z_score_deterioration(deterioration_before)
        z_scores_random = compute_z_score_deterioration(deterioration_random)

        # Print results for each dataset
        print_results(csv_file, column_means, column_means_before_nan, deterioration_before, deterioration_after, deterioration_random)
        print('-'*120)

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


    # Compute the mean across all datasets
    column_means = sum(column_means_list) / len(column_means_list)
    column_means_before_nan = sum(column_means_before_nan_list) / len(column_means_before_nan_list)
    column_means_after_nan = sum(column_means_after_nan_list) / len(column_means_after_nan_list)
    column_means_random = sum(column_means_random_list) / len(column_means_random_list)
    deteriorations_before = sum(deterioration_before_list) / len(deterioration_before_list)
    deteriorations_after = sum(deterioration_after_list) / len(deterioration_after_list)
    deterioration_random = sum(deterioration_random_list) / len(deterioration_random_list)
    missing_data_mean = sum(missing_data_list) / len(missing_data_list)

    print('#'*120)
    print(f'Number of datasets analyzed: {len(csv_files)}')
    print_results('All datasets', column_means, column_means_before_nan, deteriorations_before, deteriorations_after, deterioration_random)

    count_significant_deterioration = sum((np.abs(z_score) > 2).sum() for z_score in z_score_before_list)
    count_significant_deterioration_random = sum((np.abs(z_score) > 2).sum() for z_score in z_score_random_list)
    
    print(f'Number of significant deterioration before nan (z-score > 2): {count_significant_deterioration}')
    print(f'Compared to z-score of random missingness: {count_significant_deterioration_random}')

    # Findings (based on 3 datasets):
    # Higher tiredness, stress are indicator for missing data
    # Indicates that the missing data is not random, but more data needs to be analyzed to confirm this

    # Results (Using all datasets):
    # No observable pattern in the deterioration of the means before the missing data (for each participant and on average)
    # The z-scores are not significantly different from the z-scores that were computed from random missingness
    # -> Assume that the missing data is random

    print()
    ####################################################################################################################################

    # Print missing data percentages
    """
    #print(f'File || Missing data')
    #for i, missing_data in enumerate(missing_data_list):
    #    print(f'{csv_files[i]}: {missing_data:.2%}')
    """

    print(f'Missing data percentage mean: {missing_data_mean:.2%}')