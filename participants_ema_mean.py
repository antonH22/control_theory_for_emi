import re
import os
from ctrl import utils
import numpy as np

emas = ['EMA_mood', 'EMA_disappointed', 'EMA_scared', 'EMA_worry', 'EMA_down', 'EMA_sad', 'EMA_confidence', 'EMA_stress', 'EMA_lonely', 'EMA_energetic', 'EMA_concentration', 'EMA_resilience', 'EMA_tired', 'EMA_satisfied', 'EMA_relaxed']
emis = ['interactive1', 'interactive2', 'interactive3', 'interactive4', 'interactive5', 'interactive6', 'interactive7', 'interactive8']

rnn_model_path_MRT1 = "D:/v2_MRT1_every_valid_day"
rnn_model_path_MRT2 = "D:/v2_MRT2_every_valid_day"
rnn_model_path_MRT3 = "D:/v2_MRT3_every_valid_day"

def get_model_paths(participant_nr, folder_path):
    model_paths_dict = {}
    for filename in os.listdir(folder_path):
        match = re.search(rf'data_\d+_\d+\.csv_participant_{participant_nr}_date_(\d+\.\d+)', filename)
        if match:
            timestep = float(match.group(1))
            model_paths_dict[timestep] = os.path.join(folder_path, filename)
    return model_paths_dict

def find_model_path(file, index=None):
    # Get MRT number
    match = re.search(r"MRT(\d)", file)
    mrt_nr = int(match.group(1))
    model_paths = {1: rnn_model_path_MRT1, 2: rnn_model_path_MRT2, 3: rnn_model_path_MRT3}
    model_path = model_paths.get(mrt_nr)
    # Get participant number
    match = re.search(r'_(\d+)\.csv$', file)
    participant_nr = int(match.group(1))
    model_paths_dict = get_model_paths(participant_nr, model_path)
    if not model_paths_dict:
        return False
    if index is not None:
        # Online scenario: Choose model that is trained up to index
        valid_keys = [k for k in model_paths_dict.keys() if k <= index]
        if not valid_keys:
            return False
        key_online = max(valid_keys)  # Get the closest model up to `index`
        model_path_scenario = model_paths_dict[key_online] 
    else:
        # Offline scenario: Choose the last trained model
        max_key = max(model_paths_dict.keys())
        model_path_scenario = model_paths_dict[max_key] 
    return model_path_scenario

data_folder = "data"
subfolders = ["MRT1/processed_csv_no_con","MRT2/processed_csv_no_con","MRT3/processed_csv_no_con"]
dataset_list, files = utils.load_dataset(data_folder, subfolders, emas, emis, centered=False, exclude_constant_columns=False)

N = 50  # Number of lowest mean files to consider (also considering missing models)

means_list = []
for idx, dataset in enumerate(dataset_list):
    model_path_rnn = find_model_path(files[idx])
    if model_path_rnn == False:
        # For 33 participants there is no trained model
        print("No model found\n")
        continue

    X= dataset['X']
    mean_emas = np.nanmean(X)
    print(files[idx])
    print(mean_emas)
    print()
    means_list.append(mean_emas)

print(f'overall mean: {sum(means_list)/ len(means_list)}')

means_list.sort()

u = means_list[N - 1]
print(f"N-th lowest mean: {u}")

    


