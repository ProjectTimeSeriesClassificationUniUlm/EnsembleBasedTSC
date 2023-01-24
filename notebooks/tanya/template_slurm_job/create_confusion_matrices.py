import sys
import json
from pathlib import Path
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm
from tqdm.keras import TqdmCallback
sys.path.append("../")
from src.Evaluation import plot_model_history
from src.ModelBuilder import get_MLP, get_MCDCNN, get_Time_CNN, get_FCN, get_Encoder, get_Resnet
from src.LoadData import get_all_datasets_test_train_np_arrays
from src.Helpers import append_to_csv, get_confusion_matrix_for_model_and_data, visualize_confusion_matrix

path_to_datasets = "../datasets"

datasets_test_train_data = get_all_datasets_test_train_np_arrays(path_to_datasets)

#models_getter = [get_MLP, get_MCDCNN, get_Time_CNN, get_FCN, get_Encoder, get_Resnet]
#models_names = ["MLP", "MCDCNN", "Time_CNN", "FCN", "Encoder", "Resnet"]
models_getter = [get_Resnet, ]
models_names = ["Resnet", ]

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/opt/bwhpc/common/devel/cuda/11.8"


# tf.debugging.set_log_device_placement(True)
devices = tf.config.list_logical_devices('GPU') 
strategy = tf.distribute.MirroredStrategy(devices)
if __name__ == "__main__":
    print("Models names", models_names)
    path_persist_results = "./training_res.csv"
    if not (csv_path := Path(path_persist_results)).exists():
        csv_path.touch()
    if len(csv_path.read_text()) == 0:
        # Only write if the file is empty
        columns = ["dataset_name", "model_name", "test_loss", "test_acc", "confusion_matrix", "history"]
        append_to_csv(path_persist_results, columns)


    epochs = 30
    batch_size = 10
    validation_split = 0.2

    with strategy.scope():
        for ds_name, ds_data in tqdm(datasets_test_train_data.items(), unit='dataset'):
            print("Dataset name: ", ds_name)
            x_test, y_test = ds_data["test_data"]
            x_train, y_train = ds_data["train_data"]

            input_size = x_train.shape[1]
            output_size = len(np.unique(y_train))

            for get_model, model_name in tqdm(list(zip(models_getter, models_names)), unit='model', desc=f'Train on "{ds_name}"'):
                print("Model name: ", model_name)
                model = get_model(input_size, output_size)
                model.compile(optimizer='SGD',
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])
                history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks=[TqdmCallback(verbose=0, desc=model_name)], verbose=0)
                test_loss, test_acc = model.evaluate(x_test, y_test)

                confusion_matrix = get_confusion_matrix_for_model_and_data(model, x_test, y_test)
                #visualize_confusion_matrix(confusion_matrix, model_name, ds_name)

                row = [ds_name,
                      model_name,
                      test_loss,
                      test_acc,
                      json.dumps(confusion_matrix.tolist()), 
                      json.dumps(history.history)]
                append_to_csv(path_persist_results, row)