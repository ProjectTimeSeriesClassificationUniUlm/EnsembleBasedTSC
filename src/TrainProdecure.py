import sys
import json
from functools import reduce
from pathlib import Path
from typing import List, Callable
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm
from tqdm.keras import TqdmCallback

from Helpers import append_to_csv, get_project_root, get_confusion_matrix_for_model_and_data
from LoadData import get_all_datasets_test_train_np_arrays
from ModelBuilder import get_model_name


def train(model_builders: List[Callable],
          datasets=get_all_datasets_test_train_np_arrays("../datasets"),
          epochs=30,
          batch_size=10,
          validation_split=0.2):
    """
    Trains given models on given datasets
    :param model_builders: list of untrained models
    :param datasets: dict of datasets to be trained on
    :param epochs
    :param batch_size
    :param validation_split
    """

    # csv creation
    def model_list_to_str(models: List[Callable]):
        return reduce(lambda result, m: result + '_' + get_model_name(m), models, "")

    result_csv_path = ''.join([str(get_project_root() / "results"), "/train_",
                               model_list_to_str(model_builders), "_",
                               str(len(datasets)), "_datasets.csv"])
    if not (csv_path := Path(result_csv_path)).exists():
        csv_path.touch()
    if len(csv_path.read_text()) == 0:
        # Only write if the file is empty
        columns = ["dataset_name", "model_name", "test_loss", "test_acc", "confusion_matrix", "history"]
        append_to_csv(result_csv_path, columns)

    # model path
    model_path = str(get_project_root() / "models")

    # init tensorflow
    gpu_devices = tf.config.list_logical_devices('GPU')
    cpu_devices = tf.config.list_logical_devices('CPU')
    strategy = tf.distribute.MirroredStrategy(gpu_devices if gpu_devices else cpu_devices)
    with strategy.scope():
        # train every dataset
        for ds_name, ds_data in tqdm(datasets.items(), unit='dataset'):
            print("Dataset name: ", ds_name)
            x_test, y_test = ds_data["test_data"]
            x_train, y_train = ds_data["train_data"]

            input_size = x_train.shape[1]
            output_size = len(np.unique(y_train))

            # and every model
            for get_model in tqdm(model_builders, unit='model', desc=f'Train on "{ds_name}"'):
                model_name = get_model_name(get_model)
                print("Model name: ", model_name)
                model = get_model(input_size, output_size)
                model.compile(optimizer='adam',
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])
                history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                    validation_split=validation_split,
                                    callbacks=[TqdmCallback(verbose=0, desc=model_name)], verbose=0)
                test_loss, test_acc = model.evaluate(x_test, y_test)
                # save csv
                row = [ds_name,
                       model_name,
                       test_loss,
                       test_acc,
                       json.dumps(get_confusion_matrix_for_model_and_data(model, x_test, y_test).tolist()),
                       json.dumps(history.history)]
                append_to_csv(result_csv_path, row)
                # save model
                model_ds_path = model_path + "/" + ds_name
                Path(model_ds_path).mkdir(exist_ok=True, parents=True)
                model.save(model_ds_path + "/" + model_name + ".h5")
    return pd.read_csv(result_csv_path)
