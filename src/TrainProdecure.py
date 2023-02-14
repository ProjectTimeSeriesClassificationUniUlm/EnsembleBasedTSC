import json
from functools import reduce
from pathlib import Path

from typing import List, Callable

import keras.optimizers
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm.auto import tqdm
from tqdm.keras import TqdmCallback

from Evaluation import get_confusion_matrix_for_model_and_data
from Helpers import (
    append_to_csv,
    get_project_root,
)
from Evaluation import get_confusion_matrix_for_model_and_data
from LoadData import get_all_datasets_test_train_np_arrays
from ModelBuilder import get_model_name
from PreprocessData import preprocess_datasets


def train_single_model(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray = None,
    y_test: np.ndarray = None,
    epochs: int = 20,
    learning_rate=None,
    batch_size: int or None = 25,
    validation_split: float = 0.1,
    model_name: str = "Unnamed model",
    dataset_name: str = "Unnamed dataset",
    optimizer=keras.optimizers.Adam,
):
    if learning_rate:
        model.compile(
            optimizer=optimizer(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    else:
        model.compile(
            optimizer=optimizer(),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=[
            TqdmCallback(
                verbose=0, desc=f"Training {model_name} on {dataset_name} dataset"
            )
        ],
        verbose=0,
    )

    if x_test is not None and y_test is not None:
        test_loss, test_acc = model.evaluate(x_test, y_test)

        return model, test_loss, test_acc, history

    return model, history


def get_csv_name(used_model_names, num_datasets):
    """
    Returns a csv file name based on models and datasets

    :param used_model_names: list of model names
    :param num_datasets: number of used datasets
    :return: csv file name
    """

    def model_list_to_str(models: List[Callable]):
        return reduce(lambda result, m: result + "_" + m, models, "")

    return "".join(
        [
            str(get_project_root() / "results"),
            "/train_",
            model_list_to_str(used_model_names),
            "_",
            str(num_datasets),
            "_datasets.csv",
        ]
    )


def train(
    model_builders: List[Callable],
    datasets=None,
    epochs=30,
    batch_size=10,
    validation_split=0.2,
    optimizer=keras.optimizers.Adam,
    learning_rate=None,
    result_csv_path=None,
    unique_model_name=False,
    check_existance = False
):
    """
    Trains given models on given datasets

    :param model_builders: list of untrained models
    :param datasets: dict of datasets to be trained on
    :param epochs
    :param batch_size
    :param validation_split
    :param optimizer:
    :param learning_rate:
    :param result_csv_path:
    :param unique_model_name: appends the row index to the model name
    :return: pandas dataframe with results
    """
    if datasets is None:
        datasets = get_all_datasets_test_train_np_arrays("../datasets")
        datasets = preprocess_datasets(datasets, batch_size=batch_size)

    # csv creation
    def model_list_to_str(models: List[Callable]):
        return reduce(lambda result, m: result + "_" + get_model_name(m), models, "")

    if result_csv_path is None:
        result_csv_path = get_csv_name(
            list(map(get_model_name, model_builders)), len(datasets)
        )
    if not (csv_path := Path(result_csv_path)).exists():
        csv_path.touch()
    if len(csv_path.read_text()) == 0:
        # Only write if the file is empty
        columns = [
            "dataset_name",
            "model_name",
            "test_loss",
            "test_acc",
            "confusion_matrix",
            "history",
        ]
        append_to_csv(result_csv_path, columns)

    # model path
    model_path = str(get_project_root() / "models")

    # init tensorflow
    gpu_devices = tf.config.list_logical_devices("GPU")
    cpu_devices = tf.config.list_logical_devices("CPU")
    strategy = tf.distribute.MirroredStrategy(
        gpu_devices if gpu_devices else cpu_devices
    )
    with strategy.scope():
        # train every dataset
        for ds_name, ds_data in tqdm(datasets.items(), unit="dataset"):
            
            print("Dataset name: ", ds_name)
            x_test, y_test = ds_data["test_data"]

            input_size = ds_data["input_size"]
            output_size = ds_data["output_size"]

            # and every model
            model_index = 0
            for get_model in tqdm(
                model_builders, unit="model", desc=f'Train on "{ds_name}"'
            ):
                model_name = get_model_name(get_model) + (
                    f"-{model_index}" if unique_model_name else ""
                )
                print("Model name: ", model_name)
                model = get_model(input_size, output_size)
                
                model_ds_path = model_path + "/" + ds_name
                path_to_model_file = model_ds_path + "/" + model_name + ".h5"
                if (not check_existance) or (not Path(path_to_model_file).exists()):
                    if learning_rate:
                        model.compile(
                            optimizer=optimizer(learning_rate=learning_rate),
                            loss="sparse_categorical_crossentropy",
                            metrics=["accuracy"],
                        )
                    else:
                        model.compile(
                            optimizer=optimizer(),
                            loss="sparse_categorical_crossentropy",
                            metrics=["accuracy"],
                        )

                    history = model.fit(
                        ds_data["train_data"],
                        epochs=epochs,  # batch_size=batch_size,
                        callbacks=[TqdmCallback(verbose=0, desc=model_name)],
                        verbose=0,
                    )
                    test_loss, test_acc = model.evaluate(x_test, y_test)
                    # save csv
                    row = [
                        ds_name,
                        model_name,
                        test_loss,
                        test_acc,
                        json.dumps(
                            get_confusion_matrix_for_model_and_data(
                                model, x_test, y_test
                            ).tolist()
                        ),
                        json.dumps(history.history),
                    ]
                    append_to_csv(result_csv_path, row)
                    # save model

                    Path(model_ds_path).mkdir(exist_ok=True, parents=True)
                    model.save(path_to_model_file)
                    model_index = model_index + 1
    return pd.read_csv(result_csv_path)
