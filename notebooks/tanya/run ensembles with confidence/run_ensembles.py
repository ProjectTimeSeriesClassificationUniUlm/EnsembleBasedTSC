import logging
import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel(logging.ERROR)
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

import sys
sys.path.append("../../../src")

from toolz import keyfilter

from Evaluation import run_ensembles, load_models
from LoadData import CurrentDatasets
from Helpers import create_confusion_matrix_plot_from_csv

dataset_names = [dataset.value for dataset in CurrentDatasets][:1]

# in a nested list all elements on the same level get equal weights
# eg [foo, [bar, bar, bar, [baz, baz]]] is weighted [1/2, [1/8, 1/8, 1/8, [1/16, 1/16]]]

ensembles = {"All": [[f"MLP-{number}" for number in range(0,10)],
                     [f"Encoder-{number}" for number in range(0,10)],
                     [f"Resnet-{number}" for number in range(0,10)]],
             "MLP10": [f"MLP-{number}" for number in range(0,10)],
             "Encoder10": [f"Encoder-{number}" for number in range(0,10)],
             "Resnet10": [f"Resnet-{number}" for number in range(0,10)],
             "NNE": [[f"Resnet-{number}" for number in range(0,10)],
                     [f"Encoder-{number}" for number in range(0,10)]]}
csv_name = './ensembles_with_confidence.csv'
model_paths = '../../../models'
datasets_path = '../../../datasets'

used_ensembles=ensembles
run_ensembles(dataset_names=dataset_names, ensembles=used_ensembles, verbose=True, models_path=model_paths, datasets_path=datasets_path).to_csv(csv_name)

create_confusion_matrix_plot_from_csv(csv_name, verbose=True)