import sys
import json
from pathlib import Path
import os

import numpy as np
import math
import time
import tensorflow as tf
from tqdm.auto import tqdm
from tqdm.keras import TqdmCallback

sys.path.append("../../../src")
from Evaluation import create_confusion_matrix_plot_from_csv
from TrainProdecure import train
from PreprocessData import preprocess_data
from PreprocessData import preprocess_datasets
from LoadData import get_all_datasets_test_train_np_arrays, DATASETS_MCDCNN
from ModelBuilder import get_MLP, get_FCN, get_Resnet, get_MCDCNN, get_Encoder, get_Time_CNN, get_MCDCNN_improved

path_to_datasets = "../../../datasets"

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/opt/bwhpc/common/devel/cuda/11.8"

epochs = 30


# tf.debugging.set_log_device_placement(True)
devices = tf.config.list_logical_devices("GPU")
strategy = tf.distribute.MirroredStrategy(devices)
if __name__ == "__main__":
    with strategy.scope():
        all_datasets_names = [name for name in os.listdir(path_to_datasets) if not os.path.isdir(name)]
        datasets = get_all_datasets_test_train_np_arrays(path_to_datasets, ds_names=all_datasets_names)
        datasets = preprocess_datasets(datasets, batch_size=10, augmentation=False)
        
    tf.keras.utils.set_random_seed(math.floor(time.time()))
    df_training_res = train([get_FCN] * 10,
                        epochs=epochs,
                        datasets = datasets,
                        unique_model_name=True,
                        check_existance=True)