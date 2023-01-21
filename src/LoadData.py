import numpy as np
from sktime.datasets import load_from_tsfile_to_dataframe
import os
import warnings
from enum import Enum


class CurrentDatasets(Enum):
    abnormal_heartbeat = "abnormal_heartbeat"
    arrow_head = "arrow_head"
    coffee_beans = "coffee_beans"
    cricket_y = "cricket_y"
    distal_phalanx_outline = "distal_phalanx_outline"
    electric_devices = "electric_devices"
    freezers = "freezers"
    phalanges_outlines_correct = "phalanges_outlines_correct"
    share_price_increase = "share_price_increase"
    swedish_leaf = "swedish_leaf"
    wafer = "wafer"


def load_numpy_array_from_ts(path_to_file):
    if path_to_file is None:
        warnings.warn('path_to_file is equal to None')
        return None
    x, y = load_from_tsfile_to_dataframe(path_to_file)
    return _prepare_train_data(x), np.array(y, dtype=np.float64)


def _prepare_train_data(df):
    prepared = list()
    for j in range(df.size):
        prepared.append(df.iloc[j].to_numpy()[0].to_numpy())
    return np.array(prepared, dtype=np.float64)


def get_all_datasets_info(path_to_datasets):
    datasets_test_path_train_path = dict()
    datasets_names, datasets_paths = get_all_datasets_names_paths(path_to_datasets)
    for i in range(len(datasets_paths)):
        ds_path = datasets_paths[i]
        test_file = None
        train_file = None
        for file_name in os.listdir(ds_path):
            if "TEST" in file_name:
                test_file = os.path.join(ds_path, file_name)
            if "TRAIN" in file_name:
                train_file = os.path.join(ds_path, file_name)

        datasets_test_path_train_path[datasets_names[i]] = (test_file, train_file)
    return datasets_test_path_train_path


def get_all_datasets_names_paths(path_to_datasets):
    datasets_names = list()
    datasets_paths = list()
    for name in os.listdir(path_to_datasets):
        datasets_names.append(name)
        datasets_paths.append(os.path.join(path_to_datasets, name))
    return datasets_names, datasets_paths


def get_all_datasets_test_train_np_arrays(path_to_datasets):
    datasets_test_train_np_arrays = dict()

    datasets_test_path_train_path = get_all_datasets_info(path_to_datasets)
    for ds_name, (test_file, train_file) in datasets_test_path_train_path.items():
        x_test, y_test_raw = load_numpy_array_from_ts(test_file)
        x_train, y_train_raw = load_numpy_array_from_ts(train_file)
        lookup_table, y_train = np.unique(y_train_raw, return_inverse=True)
        lookup_table_dict = dict(zip(lookup_table, list(range(len(lookup_table)))))
        y_test = np.array(list(map(lambda label: lookup_table_dict[label], y_test_raw)))
        
        datasets_test_train_np_arrays[ds_name] = dict()
        datasets_test_train_np_arrays[ds_name]["test_data"] = x_test, y_test
        datasets_test_train_np_arrays[ds_name]["train_data"] = x_train, y_train

    return datasets_test_train_np_arrays

