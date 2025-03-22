import logging
import re
import socket
import os
import ast
from typing import Optional
from collections import defaultdict
from argparse import Namespace
import pandas as pd
from dataset.multimodal_dataset import MultimodalDataset, DatasetWrapper

log = logging.getLogger(__name__)
log.propagate = False

TIME_ANCHOR = pd.Timestamp('2022-01-01 00:00')

def split_dataframe_into_train_and_test_set(df: pd.DataFrame, obs_features: list, input_features: list,
                                  until_datetime: Optional[pd.Timestamp|str]=None, until_timestep: Optional[int]=None):
    ''' The train and test set overlap in 1 data point. It used as ground truth during 
        training and for initialization during testing. '''
    if not until_datetime in (None, 'None'):
        test_index = determine_test_index_from_timestamp(df, until_datetime)
        df_test = df.loc[test_index:]
        df_train = df.loc[:test_index+1]
        # if not isinstance(until_datetime, pd.Timestamp):   
        #     until_datetime = pd.Timestamp(until_datetime)        
        # train_timerels = (until_datetime - pd.Timestamp('2022-01-01 00:00')).total_seconds()
        # df_test = df[(df['Timerels']>=train_timerels)]
        if len(df_test)>0:
            df_train = df[df.index <= df_test.index[0]]
        else:
            df_train = df
        log.info(f'Data loaded until data {until_datetime}')
    elif not until_timestep in (None, 'None'):
        until_timestep = int(float(until_timestep))
        df_test = df.iloc[until_timestep:]
        df_train = df.iloc[:until_timestep+1]
        log.info(f'Data loaded until timestep {until_timestep}')
    else:
        df_test = df.iloc[len(df):]
        df_train = df.iloc[:]
        log.info(f'Data loaded, no test set')

    train_traj = df_train[obs_features].to_numpy()
    test_traj = df_test[obs_features].to_numpy()    
    train_inputs = df_train[input_features].to_numpy()    
    test_inputs = df_test[input_features].to_numpy()

    return train_traj, train_inputs, test_traj, test_inputs


def create_dataset_reallabor(args: Namespace|dict, data_path: Optional[str]=None, preloaded_data: Optional[pd.DataFrame]=None, 
                             verbose: Optional[str]=None, min_valid_training_timesteps: int=20):
            
    if not isinstance(args, dict):
        args_is_namespace = True
        args = vars(args)  
    else:
        args_is_namespace = False
    if data_path is not None:
        args['data_path'] = data_path
    if verbose is not None:
        args['verbose'] = verbose    
    if 'valid_ratio' in args:
        valid_ratio = args['valid_ratio']
    else:
        valid_ratio = None
    if preloaded_data is None:
        preloaded_data = pd.read_csv(args['data_path'])

    train_emas, train_inputs, test_emas, test_inputs = (
        split_dataframe_into_train_and_test_set(preloaded_data,
                                    args['obs_features'], args['input_features'],
                                    until_datetime=args['train_on_data_until_datetime'],
                                    until_timestep=args['train_on_data_until_timestep']))
    
    if not args['train_on_last_n_steps'] in (None, 'None'):
        if isinstance(args['train_on_last_n_steps'], str):
            args['train_on_last_n_steps'] = int(args['train_on_last_n_steps'])
        train_emas = train_emas[-args['train_on_last_n_steps']:]
        train_inputs = train_inputs[-args['train_on_last_n_steps']:]
    log.info(f"Loaded data with {train_emas.shape[0]} time steps.")

    if train_inputs.size == 0:
        train_inputs = None   
        test_inputs = None 
    train_dataset = MultimodalDataset('train', args['seq_len'], 
                                      ignore_leading_nans=True,
                                      return_single_tensor=False, tolerate_partial_missings=False,
                                      random_dropout_to_level=valid_ratio, verbose=args['verbose'])
    train_dataset.add_timeseries(train_emas, name='emas',
                                 feature_names=args['obs_features'])
    train_dataset.add_timeseries(train_inputs, name='inputs',
                                 feature_names=args['input_features'])
    
    if len(train_dataset.valid_indices)<min_valid_training_timesteps:
        raise RuntimeError(f"Up to date {args['train_on_data_until_datetime']}, there are only {len(train_dataset.valid_indices)} valid data points (less than 20).")

    log.debug('Successfully created train set.')
    if len(test_emas)>0:
        test_dataset = MultimodalDataset('test', 0, ignore_leading_nans=False,
                                        return_single_tensor=False,
                                        tolerate_partial_missings=False)
        try:
            test_dataset.add_timeseries(test_emas, name='emas',
                                        feature_names=args['obs_features'])
            test_dataset.add_timeseries(test_inputs, name='inputs',
                                    feature_names=args['input_features'])
        except ValueError: #There are no valid data points in test set
            log.warn(f'No valid data points in test set on date {args["train_on_data_until_datetime"]}. Test dataset will be None')
            test_dataset = None
        log.debug('Successfully created test set')
    else:
        test_dataset = None
    
    args['dim_x'] = train_dataset.timeseries['emas'].dim
    if train_inputs is not None:
        args['dim_s'] = train_dataset.timeseries['inputs'].dim
    else:
        args['dim_s'] = None
    # args['participant'] = participant_id
    if args_is_namespace:
        args = Namespace(**args)
    return args, train_dataset, test_dataset 


def create_dataset_for_hierarchized_model(args, data_dir: Optional[str]=None, preloaded_data: Optional[list[pd.DataFrame]]=None, 
                                         min_valid_training_timesteps: int=20, verbose: Optional[str]=None):
    if not isinstance(args, dict):
        args_is_namespace = True
        args = vars(args)  
    else:
        args_is_namespace = False
    
    if data_dir is not None:
        args['data_path'] = data_dir
    if verbose is not None:
        args['verbose'] = verbose  
    if preloaded_data is None:
        preloaded_data = load_all_data_from_folder(args['data_path'])
    if args['participant_subset_selector'] is not None:
        preloaded_data = select_subset_of_participants(preloaded_data, args['participant_subset_selector'])
    
    train_datasets = {}
    test_datasets = {}
    for i, df in enumerate(preloaded_data):
        participant_id = determine_participant_id(df)
        try:
            test_index = determine_test_index(participant_id, args['train_until'], args['train_test_split_row'])
        except KeyError:
            log.warning(f"Dataset {i}, participant {participant_id}, does not appear in the train/test splits. Skipping this dataset.")
            continue
        if isinstance(test_index, str):
            train_emas, train_inputs, test_emas, test_inputs = split_dataframe_into_train_and_test_set(df, args['obs_features'], args['input_features'],
                                                                                                        until_datetime=test_index)
        else:
            train_emas, train_inputs, test_emas, test_inputs = split_dataframe_into_train_and_test_set(df, args['obs_features'], args['input_features'],
                                                                                                        until_timestep=test_index)
                                                
        train_dataset = MultimodalDataset(name=f'train_{participant_id}', seq_len=args['seq_len'], ignore_leading_nans=True,
                                          return_single_tensor=False, tolerate_partial_missings=True)
        train_dataset.add_timeseries(train_emas, name='emas',
                                     feature_names=args['obs_features'])
        train_dataset.add_timeseries(train_inputs, name='inputs',
                                     feature_names=args['input_features'])
        
        if len(train_dataset.valid_indices) >= min_valid_training_timesteps:
            train_datasets[participant_id] = train_dataset
            
            if test_emas is not None and test_inputs is not None:
                log.debug(f"Creating test dataset {i}: test_emas shape: {test_emas.shape}, test_inputs shape: {test_inputs.shape}")
                test_dataset = MultimodalDataset(name=f'test_{i}', seq_len=0, 
                                                 ignore_leading_nans=False,
                                                 return_single_tensor=False,
                                                 tolerate_partial_missings=True)
                test_dataset.add_timeseries(test_emas, name='emas',
                                            feature_names=args['obs_features'])
                test_dataset.add_timeseries(test_inputs, name='inputs',
                                            feature_names=args['input_features'])
                log.debug(f"Test Dataset {i}: Valid indices: {len(test_dataset.valid_indices)}, Valid sequence indices: {len(test_dataset.valid_sequence_indices)}")
                test_datasets[participant_id] = test_dataset
        else:
            log.warning(f"Dataset {i}, participant {participant_id}, has only {len(train_dataset.valid_indices)} valid data points (less than {min_valid_training_timesteps}). Skipping this dataset.")

    if not train_datasets:
        raise RuntimeError(f"No valid datasets found with at least {min_valid_training_timesteps} valid data points.")

    log.info(f'Successfully created {len(train_datasets)} train sets and {len(test_datasets)} test sets.')

    train_dataset_wrapper = DatasetWrapper(train_datasets)
    test_dataset_wrapper = DatasetWrapper(test_datasets) if len(test_datasets)>0 else None

    if 'dim_x' not in args:
        args['dim_x'] = train_dataset_wrapper.datasets.iloc[0].timeseries['emas'].dim
    elif args['dim_x'] != train_dataset_wrapper.datasets.iloc[0].timeseries['emas'].dim:
        log.warning(f"Original argument 'dim_x' is {args['dim_x']} but the loaded dataset has {train_dataset_wrapper.datasets.iloc[0].timeseries['emas'].dim} observation dimensions.")
    if 'dim_s' not in args:
        args['dim_s'] = train_dataset_wrapper.datasets.iloc[0].timeseries['inputs'].dim
    elif args['dim_s'] != train_dataset_wrapper.datasets.iloc[0].timeseries['inputs'].dim:
        log.warning(f"Original argument 'dim_a' is {args['dim_a']} but the loaded dataset has {train_dataset_wrapper.datasets.iloc[0].timeseries['inputs'].dim} input dimensions.")
    if 'subject_indices' not in args:
        args['subject_indices'] = train_dataset_wrapper.datasets.index.to_list()
    elif args['subject_indices'] != train_dataset_wrapper.datasets.index.to_list():
        log.warning(f"Original argument 'subject_indices' is {args['subject_indices']} but the loaded dataset has indices {train_dataset_wrapper.datasets.index.to_list()}.")
    if 'n_subjects' not in args:
        args['n_subjects'] = len(train_dataset_wrapper.datasets)
    elif args['n_subjects'] != len(train_dataset_wrapper.datasets):
        log.warning(f"Original argument 'n_subjects' is {args['n_subjects']} but the loaded dataset has {len(train_dataset_wrapper.datasets)} subjects.")

    if args_is_namespace:
        args = Namespace(**args)

    return args, train_dataset_wrapper, test_dataset_wrapper


def join_base_path(*join_with_path):
    hostname = socket.gethostname()
    if hostname=='5CD204H4Z5':    # hp, local machine
        if 'reallaborai4u' in join_with_path[0]:
            base_path = 'D:/ZI Mannheim'
        else:
            base_path = 'D:/ZI Mannheim/KI Reallabor'
    else:
        base_path = '/home/janik.fechtelpeter/Documents'
    return os.path.join(base_path, *join_with_path)

def join_ordinal_bptt_path(*join_with_path):
    return join_base_path(os.path.join('ordinal-bptt', *join_with_path))


def data_management_path(MRT: int):
    if MRT==1:
        path = join_base_path('reallaborai4u/data_management')
    elif MRT==2:
        path = join_base_path('reallaborai4u/data_management_MRT2')
    elif MRT==3:
        path = join_base_path('reallaborai4u/data_management_MRT3')
    else:
        raise ValueError(f'MRT {MRT} is not a valid MRT.')
    return path

def dataset_path(MRT: int, dataset_name: str):
    return os.path.join(data_management_path(MRT), dataset_name)

def train_test_split_path(MRT: int, split_file: str):
    return os.path.join(data_management_path(MRT), 'train_test_splits', split_file)

def get_data_file(MRT: int, participant: int, dataset_name: str):
    if MRT>1:
        study_id = '12600'
    else:
        study_id = '11228'
    data_file = os.path.join(dataset_path(MRT, dataset_name), f'{study_id}_{participant}.csv')
    return data_file

def easy_reallabor_dataset(MRT: int, participant: int, dataset_name: str, train_on_data_until_timestep: int):
    
    args = {'data_path': get_data_file(MRT, participant, dataset_name),
            'train_on_data_until_timestep': train_on_data_until_timestep,
            'train_on_data_until_datetime': None,
            'obs_features': ['EMA_mood','EMA_disappointed','EMA_scared','EMA_worry',
                              'EMA_down','EMA_sad','EMA_confidence','EMA_stress','EMA_lonely',
                              'EMA_energetic','EMA_concentration','EMA_resilience','EMA_tired',
                              'EMA_satisfied', 'EMA_relaxed'],
            'input_features': ['interactive1', 'interactive2', 'interactive3', 'interactive4',
                               'interactive5', 'interactive6','interactive7', 'interactive8'],
            'train_on_last_n_steps': None,
            'seq_len': 50,
            'batch_size': 1,
            'batches_per_epoch': 0,
            'verbose': 'none'}
    return create_dataset_reallabor(args)


def determine_test_index(participant_id: int, train_until: Optional[int|str|dict], train_test_split_row: Optional[int] = None) -> int|str|None:   
    # if train_until is a path or a dict, read the value belonging to the participant_id
    if isinstance(train_until, str):
        # path encoded in string
        if os.path.exists(train_until):
            test_index_df = pd.read_csv(train_until, index_col=0, header=0)
            if train_test_split_row is not None:
                if isinstance(train_test_split_row, str):
                    train_test_split_row = int(train_test_split_row)
                train_until = int(test_index_df[str(participant_id)].iloc[train_test_split_row])
            else:
                raise ValueError('train_test_split_row must be provided if train_until is a path to a csv file.')
        # dict encoded in string
        else:
            evaluated_train_until_string = eval_literal_or_none(train_until)
            if isinstance(evaluated_train_until_string, dict):
                train_until = evaluated_train_until_string[participant_id]
    # literal dict
    elif isinstance(train_until, dict):
        train_until = train_until[participant_id]
    
    # once value is read from file or dict, train_until can be a timestamp, int, or None, or a string encoding for any of these
    if isinstance(train_until, str):
        if is_date_string(train_until):
            return train_until
        elif is_int_string(train_until):
            return int(train_until)
        elif train_until.lower() == 'none':
            return None
        else:
            raise ValueError(f'"train_until" value "{train_until}" is an uninterpretable string. Please provide a valid string representing a date or an integer.')
    elif isinstance(train_until, int):
        return train_until
    elif train_until is None:
        return None
    else:
        raise ValueError(f'"train_until" value "{train_until}" is not a string or integer.')

def determine_test_index_from_timestamp(df: pd.DataFrame, timestamp: str|pd.Timestamp) -> int:
    if not isinstance(timestamp, pd.Timestamp):
        timestamp = pd.Timestamp(timestamp)
    train_timerels = (timestamp - TIME_ANCHOR).total_seconds()
    test_index = df.index[(df['Timerels']>=train_timerels)][0]
    return test_index

def select_subset_of_participants(data_list: list[pd.DataFrame], selector: list[int]|str):

    if isinstance(selector, str):
        if os.path.exists(selector):        # selector is a path to a train test split file
            selector_df = pd.read_csv(selector, index_col=0, header=0)
            subset = [int(c) for c in selector_df.columns]
        else:
            subset = eval_literal_or_none(selector)
    
    elif isinstance(selector, list):
        subset = selector
    
    data_list = [d for d in data_list if determine_participant_id(d) in subset]
    return data_list

def is_date_string(string: str):
    try:
        pd.Timestamp(string)
        return True
    except (ValueError, TypeError):
        return False
    
def is_int_string(string: str):
    try:
        int(string)
        return True
    except (ValueError, TypeError):
        return False
    
def eval_literal_or_none(string: str):
    try:
        return ast.literal_eval(string)
    except:
        return None
    
def determine_participant_id(df: pd.DataFrame):
    if 'participant' in df.columns:
        return df['participant'].iloc[0]
    elif 'Participant' in df.columns:
        return df['Participant'].iloc[0]
    else:
        raise ValueError('No participant column found in the data.')
    
def load_all_data_from_folder(folder: str) -> list[pd.DataFrame]:   
    preloaded_data = []
    for i, filename in enumerate(os.listdir(folder)):
        if filename.endswith('.csv'):
            preloaded_data.append(pd.read_csv(os.path.join(folder, filename)))
    return preloaded_data