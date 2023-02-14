import logging
import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel(logging.ERROR)
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

import sys
sys.path.append("../src")

from toolz import keyfilter

from Evaluation import create_confusion_matrix_plot_from_csv
from LoadData import CurrentDatasets
from EnsembleBuilder import EnsembleBuilder

path_to_datasets = "../../../datasets"
path_to_models = "../../../models"

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/opt/bwhpc/common/devel/cuda/11.8"

epochs = 30

devices = tf.config.list_logical_devices("GPU")
strategy = tf.distribute.MirroredStrategy(devices)
if __name__ == "__main__":
    with strategy.scope():
        
        dataset_names = [name for name in os.listdir("../datasets") if not os.path.isdir(name)] # all datasets
        # in a nested list all elements on the same level get equal weights
        # eg [foo, [bar, bar, bar, [baz, baz]]] is weighted [1/2, [1/8, 1/8, 1/8, [1/16, 1/16]]]

        ensembles = {"All": [[f"MLP-{number}" for number in range(0,10)],
                             [f"FCN-{number}" for number in range(0,10)],
                             [f"MCDCNN_improved-{number}" for number in range(0,10)],
                             [f"Encoder-{number}" for number in range(0,10)],
                             [f"Resnet-{number}" for number in range(0,10)]],
                     "MLP10": [f"MLP-{number}" for number in range(0,10)],
                     "FCN10": [f"FCN-{number}" for number in range(0,10)],
                     "MCDCNN_improved10": [f"MCDCNN_improved-{number}" for number in range(0,10)],
                     "Encoder10": [f"Encoder-{number}" for number in range(0,10)],
                     "Resnet10": [f"Resnet-{number}" for number in range(0,10)],
                     "NNE": [[f"Resnet-{number}" for number in range(0,10)],
                             [f"FCN-{number}" for number in range(0,10)],
                             [f"Encoder-{number}" for number in range(0,10)]],
                     "Best4": [[f"Resnet-{number}" for number in range(0,10)],
                               [f"FCN-{number}" for number in range(0,10)],
                               [f"MLP-{number}" for number in range(0,10)],
                               [f"Encoder-{number}" for number in range(0,10)]]}
        csv_name = './ensembles_without_augmentation_all_datasets.csv'
        used_ensembles=ensembles
        
        ens_builder = EnsembleBuilder(dataset_names=dataset_names, 
                              ensembles=used_ensembles, 
                              verbose=True,
                              models_path="../models/",
                              datasets_path="../datasets/")
        ens_builder.run_ensembles(augmentation=False).to_csv(csv_name)
        
        create_confusion_matrix_plot_from_csv(csv_name, verbose=True)
        