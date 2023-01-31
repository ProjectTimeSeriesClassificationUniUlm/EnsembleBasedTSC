import time
import math
import os
import logging
import keras.optimizers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel(logging.WARNING)

import json
import sys
sys.path.append("../../../src")
from Helpers import visualize_confusion_matrix, create_confusion_matrix_plot_from_csv
from Evaluation import plot_model_history
from TrainProdecure import train
from ModelBuilder import get_MLP, get_FCN, get_Resnet, get_MCDCNN, get_Encoder, get_Time_CNN
from LoadData import DATASETS_MCDCNN, DATASETS_TIME_CNN, DATASETS_FCN, DATASETS_ENCODER, DATASETS_RESNET, get_all_datasets_test_train_np_arrays

path_to_datasets = "../../../datasets"

datasets_test_train_data = get_all_datasets_test_train_np_arrays(path_to_datasets)
datasets_test_train_data = {k:v for k, v in datasets_test_train_data.items() if k in DATASETS_RESNET}

#models_getter = [get_MLP, get_MCDCNN, get_Time_CNN, get_FCN, get_Encoder, get_Resnet]
#models_names = ["MLP", "MCDCNN", "Time_CNN", "FCN", "Encoder", "Resnet"]
models_getter = [get_Resnet, ]
models_names = ["Resnet", ]

#optimizers = [keras.optimizers.Adam, keras.optimizers.SGD]
#learning_rates = [0.01, 0.001]
optimizers = [keras.optimizers.SGD, ]
learning_rates = [0.001, ]

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/opt/bwhpc/common/devel/cuda/11.8"

epochs = 30
tf.keras.utils.set_random_seed(math.floor(time.time()))

# tf.debugging.set_log_device_placement(True)
devices = tf.config.list_logical_devices('GPU') 
strategy = tf.distribute.MirroredStrategy(devices)
if __name__ == "__main__":

    for optimizer, optimizer_name in zip([keras.optimizers.Adam, keras.optimizers.SGD], ["adam", "sgd"]):
        for learning_rate in learning_rates:
            train(result_csv_path=f"./opt_{optimizer_name}_lr_{str(learning_rate)}.csv",
                  datasets=datasets_test_train_data,
                 learning_rate = learning_rate,
                 optimizer = optimizer,
                 model_builders = models_getter)
        