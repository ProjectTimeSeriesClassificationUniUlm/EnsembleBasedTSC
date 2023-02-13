import numpy as np
from sktime.datasets import load_from_tsfile_to_dataframe
import os
import warnings
from enum import Enum


# Datasets work for "MLP", "FCN", "Encoder", "Resnet" models
class CurrentDatasets(str, Enum):
    cricket_y = "cricket_y"
    distal_phalanax_tw = "distal_phalanax_tw"
    egg_five_days = "egg_five_days"
    electric_devices = "electric_devices"
    face_ucr = "face_ucr"
    fifty_words = "fifty_words"
    mote_strain = "mote_strain"
    power_cons = "power_cons"
    sony_robot = "sony_robot"
    swedish_leaf = "swedish_leaf"
    synthetic_control = "synthetic_control"


DATASETS_MLP = [
    "cbf",
    "car",
    "distal_phalanax_tw",
    "swedish_leaf",
    "electric_devices",
    "fifty_words",
    "freezers",
    "synthetic_control",
    "trace",
    "fish",
    "lightning_7",
    "cricket_y",
    "coffee_beans",
    "bme",
    "plane",
    "power_cons",
    "face_ucr",
    "medical_images",
    "arrow_head",
    "sony_robot",
    "mote_strain",
    "egg_five_days",
]

DATASETS_MCDCNN = ["electric_devices", "freezers", "sony_robot"]

DATASETS_TIME_CNN = ["electric_devices", "mote_strain", "egg_five_days"]

DATASETS_FCN = [
    "cbf",
    "gun_point_male_female",
    "distal_phalanax_tw",
    "swedish_leaf",
    "distal_phalanx_outline",
    "strawberry",
    "electric_devices",
    "fifty_words",
    "synthetic_control",
    "cricket_y",
    "gun_point_old_young",
    "power_cons",
    "face_ucr",
    "sony_robot",
    "mote_strain",
    "egg_five_days",
    "large_kitchen_appliances",
]

DATASETS_ENCODER = [
    "cbf",
    "gun_point_male_female",
    "car",
    "distal_phalanax_tw",
    "phalanges_outlines_correct",
    "swedish_leaf",
    "distal_phalanx_outline",
    "strawberry",
    "electric_devices",
    "fifty_words",
    "freezers",
    "synthetic_control",
    "gun_point_age_span",
    "trace",
    "fish",
    "lightning_7",
    "cricket_y",
    "coffee_beans",
    "gun_point_old_young",
    "middle_phalanx_correct",
    "bme",
    "plane",
    "power_cons",
    "face_ucr",
    "medical_images",
    "arrow_head",
    "sony_robot",
    "mote_strain",
    "egg_five_days",
]

DATASETS_RESNET = [
    "gun_point_male_female",
    "distal_phalanax_tw",
    "swedish_leaf",
    "distal_phalanx_outline",
    "strawberry",
    "electric_devices",
    "fifty_words",
    "synthetic_control",
    "cricket_y",
    "gun_point_old_young",
    "middle_phalanx_correct",
    "power_cons",
    "face_ucr",
    "medical_images",
    "sony_robot",
    "mote_strain",
    "egg_five_days",
    "large_kitchen_appliances",
]


def get_mlp_datasets_test_train_np_arrays(path_to_datasets):
    return get_all_datasets_test_train_np_arrays(
        path_to_datasets, ds_names=DATASETS_MLP
    )


def get_mcdcnn_datasets_test_train_np_arrays(path_to_datasets):
    return get_all_datasets_test_train_np_arrays(
        path_to_datasets, ds_names=DATASETS_MCDCNN
    )


def get_time_cnn_datasets_test_train_np_arrays(path_to_datasets):
    return get_all_datasets_test_train_np_arrays(
        path_to_datasets, ds_names=DATASETS_TIME_CNN
    )


def get_fcn_datasets_test_train_np_arrays(path_to_datasets):
    return get_all_datasets_test_train_np_arrays(
        path_to_datasets, ds_names=DATASETS_FCN
    )


def get_encoder_datasets_test_train_np_arrays(path_to_datasets):
    return get_all_datasets_test_train_np_arrays(
        path_to_datasets, ds_names=DATASETS_ENCODER
    )


def get_resnet_datasets_test_train_np_arrays(path_to_datasets):
    return get_all_datasets_test_train_np_arrays(
        path_to_datasets, ds_names=DATASETS_RESNET
    )


def get_all_datasets_test_train_np_arrays(path_to_datasets, ds_names=None):
    datasets_test_train_np_arrays = dict()

    datasets_test_path_train_path = get_all_datasets_info(path_to_datasets, ds_names)
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


def get_all_datasets_info(path_to_datasets, ds_names=None):
    datasets_test_path_train_path = dict()
    datasets_names, datasets_paths = get_all_datasets_names_paths(
        path_to_datasets, ds_names
    )
    for i in range(len(datasets_paths)):
        ds_path = datasets_paths[i]
        test_file = None
        train_file = None
        for file_name in os.listdir(ds_path):
            if "TEST" in file_name:
                test_file = os.path.join(ds_path, file_name)
            if "TRAIN" in file_name:
                train_file = os.path.join(ds_path, file_name)
        if test_file and train_file:
            datasets_test_path_train_path[datasets_names[i]] = (test_file, train_file)
    return datasets_test_path_train_path


def get_all_datasets_names_paths(path_to_datasets, ds_names=None):
    datasets_names = list()
    datasets_paths = list()
    for name in os.listdir(path_to_datasets):
        if ds_names is None or name in ds_names:
            datasets_names.append(name)
            datasets_paths.append(os.path.join(path_to_datasets, name))
    return datasets_names, datasets_paths


def load_numpy_array_from_ts(path_to_file):
    if path_to_file is None:
        warnings.warn("path_to_file is equal to None")
        return None
    x, y = load_from_tsfile_to_dataframe(path_to_file)
    return _prepare_train_data(x), np.array(y, dtype=np.float64)


def _prepare_train_data(df):
    prepared = list()
    for j in range(df.size):
        prepared.append(df.iloc[j].to_numpy()[0].to_numpy())
    return np.array(prepared, dtype=np.float64)
