from ctrl import utils
import os
import glob

# Preprocess data and save in preprocessed_data folder
emas = ['EMA_mood', 'EMA_disappointed', 'EMA_scared', 'EMA_worry', 'EMA_down', 'EMA_sad', 'EMA_confidence', 'EMA_stress', 'EMA_lonely', 'EMA_energetic', 'EMA_concentration', 'EMA_resilience', 'EMA_tired', 'EMA_satisfied', 'EMA_relaxed']
emis = ['interactive1', 'interactive2', 'interactive3', 'interactive4', 'interactive5', 'interactive6', 'interactive7', 'interactive8']
invert_columns = ['EMA_mood', 'EMA_confidence', 'EMA_energetic', 'EMA_concentration', 'EMA_resilience', 'EMA_satisfied', 'EMA_relaxed']

input_folder = "data/MRT1/processed_csv_no_con"
output_folder = "prep_data/MRT1"

# Get all CSV files in the input folder
csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

# Process each CSV file and save in the output folder
for csv_file in csv_files:
    filename = os.path.basename(csv_file)
    # Select the emas and emis and remove missing rows at the beginning
    data = utils.csv_to_dataset(csv_file, emas, emis, invert_columns, regularize=True)
    filename = filename.replace(".csv", "_prep.csv")
    output_path = os.path.join(output_folder, filename)
    # Save the preprocessed data to a CSV file
    utils.dataset_to_csv(data, emas, emis, output_path)

print("Processing complete!")