import pandas as pd
import numpy as np
from scipy.stats import shapiro

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

# The datasets need to be preprocessed before using this code (select columns, invert columns)
# Load the datasets
df1 = pd.read_csv("Path")
df2 = pd.read_csv("Path")
df3 = pd.read_csv("Path")

column_means1 = compute_mean_without_nan(df1, emas)
column_means2 = compute_mean_without_nan(df2, emas)
column_means3 = compute_mean_without_nan(df3, emas)

column_means_nan1 = compute_mean_with_nan_infront(df1, emas)
column_means_nan2 = compute_mean_with_nan_infront(df2, emas)
column_means_nan3 = compute_mean_with_nan_infront(df3, emas)

deterioration1 = compute_deterioration_means(column_means1, column_means_nan1)
deterioration2 = compute_deterioration_means(column_means2, column_means_nan2)
deterioration3 = compute_deterioration_means(column_means3, column_means_nan3)

column_means = (column_means1 + column_means2 + column_means3) / 3.0
column_means_nan = (column_means_nan1 + column_means_nan2 + column_means_nan3) / 3.0
deteriorations = (deterioration1 + deterioration2 + deterioration3) / 3.0

# Perform the Shapiro-Wilk Test for normality
check_aprox_normal = np.array(deteriorations)
statistic, p_value = shapiro(check_aprox_normal)

# Print the results
#print(f"Shapiro-Wilk Test Statistic: {statistic}")
#print(f"p-value: {p_value}")
#The p-value (0.2288) is greater than 0.05-> the data is approximately normal

z_scores = compute_z_score_deterioration(deteriorations)

#positive deterioration: worsend mental health before missing data
concatenated = pd.concat([column_means, column_means_nan, deteriorations, z_scores], axis=1)
concatenated.columns = ['means', 'means before missing data','deterioration','z_score']
print(concatenated)

# Sum over all deteriorations (positive deterioration: worsend mental health before missing data)
mean_deterioration = deteriorations.sum() / len(emas)
print(f'Deterioration mean: {mean_deterioration}')

# Findings:
# Higher tiredness, stress are indicator for missing data
# Indicates that the missing data is not random, but more data needs to be analyzed to confirm this
# -> Informative missingness

####################################################################################################################################
# Missing data percentage
def compute_missing_data_percentage(df):
    df = df[emas]
    rows_with_missing_data = df[df.isna().any(axis=1)].index
    num_rows_with_missing_data = len(rows_with_missing_data)
    return num_rows_with_missing_data / (len(df) -1)
    
missing_data1 = compute_missing_data_percentage(df1)
missing_data2 = compute_missing_data_percentage(df2)
missing_data3 = compute_missing_data_percentage(df3)

print()
print(f'Missing data percentage 1: {missing_data1}')
print(f'Missing data percentage 2: {missing_data2}')
print(f'Missing data percentage 3: {missing_data3}')

missing_data = (missing_data1 + missing_data2 + missing_data3) / 3.0
print(f'Missing data percentage mean: {missing_data}')