import pandas as pd
import numpy as np
from scipy.stats import shapiro
import os
import glob

from collections import Counter

emas = ['EMA_mood', 'EMA_disappointed', 'EMA_scared', 'EMA_worry', 'EMA_down', 'EMA_sad', 'EMA_confidence', 'EMA_stress', 'EMA_lonely', 'EMA_energetic', 'EMA_concentration', 'EMA_resilience', 'EMA_tired', 'EMA_satisfied', 'EMA_relaxed', 'EMA_sleep', 'EMA_joyful_day', 'EMA_feelactive_sincebeep', 'EMA_activity_pleas', 'EMA_social_satisfied', 'EMA_social_alone_yes', 'EMA_firstsignal', 'Positive affect', 'Negative affect', 'Self-esteem', 'Worrying', 'Activity level', 'Stress', 'Social isolation']

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
    """
    # Ensure each selected row has at least 3 preceding rows without NaN
    valid_indices = [
        idx for idx in indices_before_missing 
        if (idx >= 3 and df.iloc[idx-3:idx].notna().all().all())
    ]
    rows_before_nan2 = df.loc[valid_indices]
    """
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


def bootstrap_ci(data, func=np.mean, n_bootstrap=1000, ci=95, seed=None):
    """
    Computes a bootstrap confidence interval for a given statistic.

    Parameters:
    - data (array-like): The sample data.
    - func (callable): The statistic function (default: np.mean).
    - n_bootstrap (int): Number of bootstrap resamples.
    - ci (float): Confidence level (default: 95%).
    - seed (int, optional): Random seed for reproducibility.

    Returns:
    - tuple: (lower bound, upper bound) of the confidence interval.
    """
    if seed is not None:
        np.random.seed(seed)
    
    bootstrap_samples = np.array([func(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)])
    lower, upper = np.percentile(bootstrap_samples, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return lower, upper

# Apply bootstrapping to each column in the dataframe (each EMA variable)
def apply_bootstrapping(df, n_bootstrap=1000, ci=95, seed=None):
    bootstrap_results = {}

    for ema in df.columns:
        data = df[ema].values
        lower, upper = bootstrap_ci(data, n_bootstrap=n_bootstrap, ci=ci, seed=seed)
        bootstrap_results[ema] = [lower, upper]

    # Convert the results to a DataFrame for better readability
    df_bootstrap = pd.DataFrame.from_dict(bootstrap_results, orient='index', columns=['CI_Lower', 'CI_Upper'])

    return df_bootstrap

def print_results(csv_file, column_means, column_means_before_nan, deteriorations_before, deteriorations_after, deterioration_random, bootstrap_df, bootstrap_random_df):
    # Concatenate results into a DataFrame
    z_scores_before = compute_z_score_deterioration(deteriorations_before)
    z_scores_after = compute_z_score_deterioration(deteriorations_after)
    z_scores_random = compute_z_score_deterioration(deterioration_random) 

    print(f'{csv_file}:')
    concatenated = pd.concat([column_means, column_means_before_nan, deteriorations_before, significant_bootstrap, z_scores_before, z_scores_after, z_scores_random], axis=1)
    concatenated.columns = ['means', 'means b. nan', 'deteriorations b. nan','significant (bootstr.)','z_scores', 'z_scores (after nan)', 'z_scores (random)']
    print(concatenated)
    print()
    
    print("Bootstrapping confidence intervals for deterioration values:")
    # Join the bootstrap results with the original deterioration values
    df_deterioration = pd.DataFrame({"Deterioration": deteriorations_before})
    # Display the final table
    # Rename columns in bootstrap_random_df
    bootstrap_random_df = bootstrap_random_df.rename(columns={"CI_Lower": "CI_Lower_random", "CI_Upper": "CI_Upper_random"})
    df_bootstrap_results = df_deterioration.join(bootstrap_df).join(bootstrap_random_df)
    print(df_bootstrap_results)

def print_correct_overall_results(column_means, column_means_before_nan, column_means_before_nan_correct, deteriorations_before, correct_deterioration_before, deterioration_random, overall_deteriorations_df):
    z_scores_correct = compute_z_score_deterioration(correct_deterioration_before)
    z_scores_random = compute_z_score_deterioration(deterioration_random)

    df_bootstrap = apply_bootstrapping(overall_deteriorations_df, n_bootstrap=10000, seed=42)
    # Bootstrap confidence intervals for all deterioration values per ema (if the confidence interval does not include 0 the deterioration is significantly different from 0)
    significant_bootstrap = (df_bootstrap["CI_Lower"] > 0) | (df_bootstrap["CI_Upper"] < 0)

    concatenated = pd.concat([column_means, column_means_before_nan_correct, correct_deterioration_before, significant_bootstrap, z_scores_correct, deterioration_random, z_scores_random], axis=1)
    concatenated.columns = ['means', 'means nan', 'deterioration','significant (bootstr.)', 'z-score (correct)','deteriorations random', 'z-score (random)']
    print(concatenated)
    print()
    print("Bootstrapping confidence intervals for deterioration values:")
    # Join the bootstrap results with the original deterioration values
    df_deterioration = pd.DataFrame({"Deterioration": correct_deterioration_before})
    # Display the final table
    df_bootstrap_results = df_deterioration.join(df_bootstrap)
    print(df_bootstrap_results)
    print('-'*120)
    
def number_rows_before_nan(file):
    # Get indices immediately BEFORE missing data
    df = pd.read_csv(file)
    df = df[emas]
    missing_rows = df.isna().any(axis=1)
    start_of_missing = missing_rows & (~missing_rows.shift(1, fill_value=False))
    indices_before_missing = start_of_missing[start_of_missing].index - 1
    
    """
    # Ensure each selected row has at least 1 preceding rows without NaN
    # To exclude files where a person just answered every second EMA
    valid_indices = [
        idx for idx in indices_before_missing 
        if (idx >= 1 and df.iloc[idx-1:idx].notna().all().all())
    ]
    """
    return len(indices_before_missing)

###########################################################################################################################################
###########################################################################################################################################
# This block will only run if the script is executed directly
if __name__ == "__main__":
    # Get all CSV files in the preprocessed data folder
    data_folder = "data"
    subfolders = ["MRT1/processed_csv_no_con","MRT2/processed_csv_no_con","MRT3/processed_csv_no_con"]

    # Collect all CSV files with less than 50% missing rows from the specified subfolders
    csv_files = []
    """
    # Analyse MRT1-3 combined
    for subfolder in subfolders:
        folder_path = os.path.join(data_folder, subfolder, "*.csv")
        for file in glob.glob(folder_path):
            num_before_missing = number_rows_before_nan(file)
            if num_before_missing > 20:
                csv_files.append(file)
    """
    
    # Analyse MRT1-3 seperate:
    mrt_nr = 1

    folder_path = os.path.join(data_folder, subfolders[mrt_nr-1], "*.csv")
    for file in glob.glob(folder_path):
        num_before_missing = number_rows_before_nan(file)
        if num_before_missing > 20:
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
    overall_significant_emas_list = []
    overall_significant_emas_random_list = []

    overall_num_alone_ratio = []
    overall_num_alone_before_nan_ratio = []

    # Process each CSV file
    for i,csv_file in enumerate(csv_files):
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
        bootstrap_df = apply_bootstrapping(deteriorations_df, seed=42)
        significant_bootstrap = (bootstrap_df["CI_Lower"] > 0) | (bootstrap_df["CI_Upper"] < 0)
        significant_emas = bootstrap_df.index[significant_bootstrap].tolist()
        overall_significant_emas_list.extend(significant_emas)
        num_significant_bootstrap = significant_bootstrap.sum()
        overall_num_significant_bootstrap += num_significant_bootstrap

        # Bootstrap confidence intervals for random deterioration values
        bootstrap_random_df = apply_bootstrapping(random_deteriorations_df, seed=42)
        significant_bootstrap_random = (bootstrap_random_df["CI_Lower"] > 0) | (bootstrap_random_df["CI_Upper"] < 0)
        significant_emas_random = bootstrap_random_df.index[significant_bootstrap_random].tolist()
        overall_significant_emas_random_list.extend(significant_emas_random)
        num_significant_bootstrap_random = significant_bootstrap_random.sum()
        overall_num_significant_bootstrap_random += num_significant_bootstrap_random

        # Print results for each dataset
        #print_results(csv_file, column_means, column_means_before_nan, deterioration_before, deterioration_after, deterioration_random, bootstrap_df, bootstrap_random_df)
        print(f'{csv_file}:')
        print(f'Number of significant deterioration before nan (bootstrap): {num_significant_bootstrap}')
        print(f'Emas with significant deterioration: {significant_emas}')
        print(f'Compared to number of random missingness: {num_significant_bootstrap_random}')
        print(f'Emas with significant deterioration random: {significant_emas_random}')
        print('-'*120)
        # Convert to DataFrame

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

###########################################################################################################################################################################
    # Compute the mean across all datasets
    column_means = sum(column_means_list) / len(column_means_list)
    column_means_before_nan = sum(column_means_before_nan_list) / len(column_means_before_nan_list)
    column_means_before_nan_correct = rows_before_nan_overall_df.mean()
    correct_deterioration_before= compute_deterioration_means(column_means, column_means_before_nan_correct)
    column_means_after_nan = sum(column_means_after_nan_list) / len(column_means_after_nan_list)
    column_means_random = sum(column_means_random_list) / len(column_means_random_list)
    deteriorations_before = sum(deterioration_before_list) / len(deterioration_before_list)
    deteriorations_after = sum(deterioration_after_list) / len(deterioration_after_list)
    deterioration_random = compute_deterioration_means(column_means, random_rows_df.mean())
    missing_data_mean = sum(missing_data_list) / len(missing_data_list)
    
    print('#'*120)
    print(f"MRT{mrt_nr}")
    print(f'Number of datasets analyzed: {len(csv_files)}')
    print_correct_overall_results(column_means, column_means_before_nan, column_means_before_nan_correct, deteriorations_before, correct_deterioration_before, deterioration_random, overall_deteriorations_df)

    count_significant_deterioration = sum((np.abs(z_score) > 2).sum() for z_score in z_score_before_list)
    count_significant_deterioration_random = sum((np.abs(z_score) > 2).sum() for z_score in z_score_random_list)

    print(f'Number of significant deterioration before nan (bootstrap): {overall_num_significant_bootstrap}')
    print(f'Compared to number of random missingness: {overall_num_significant_bootstrap_random}')
    
    ema_counts = Counter(overall_significant_emas_list)
    ema_counts_random = Counter(overall_significant_emas_random_list)
    df_significant_counts = pd.DataFrame.from_dict(
        {"Occurences": ema_counts, "Occurences random": ema_counts_random},
    ).fillna(0)
    df_counts_sorted = df_significant_counts.sort_values(by='Occurences', ascending=False)
    print(df_counts_sorted)
    
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

    # Findings: 
    # - No general indicator for missing data was found, but individual indicators can be found for some participants.
