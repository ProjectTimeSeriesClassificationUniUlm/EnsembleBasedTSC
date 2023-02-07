import json
import os
from functools import partial

import numpy as np
from itertools import combinations
import tensorflow_addons  # needed for model import, do not remove
import pandas
from keras.callbacks import LambdaCallback
from matplotlib import pyplot as plt
from numpy import array_equal
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow import keras
from toolz import thread_first, thread_last, identity, count, groupby

from Ensemble import EnsembleMethods, Ensemble
from LoadData import CurrentDatasets, get_all_datasets_test_train_np_arrays
from PreprocessData import preprocess_datasets, add_additive_white_gaussian_noise
from Helpers import remove_suffix


def plot_model_history(history, epochs=None, path_to_persist=None):
    if epochs is None:
        print("No epochs specified, using all epochs.")
        epochs = len(history['loss'])

    else:
        print(f"Using {epochs} epochs. You can ignore the epochs parameter if you want to use all epochs.")

    loss = history['loss']
    val_loss = history['val_loss']

    accuracy = history['accuracy']
    val_accuracy = history['val_accuracy']
    eps = range(epochs)

    figure, axis = plt.subplots(2, 1)
    axis[0].plot(eps, loss, 'r', label='Training loss')
    axis[0].plot(eps, val_loss, 'b', label='Validation loss')
    axis[0].set_title('Training and Validation Loss')
    axis[0].set_xlabel('Epoch')
    axis[0].set_ylabel('Loss Value')
    axis[0].set_ylim([0, 2])
    axis[0].legend()

    axis[1].plot(eps, accuracy, 'r', label='Training accuracy')
    axis[1].plot(eps, val_accuracy, 'b', label='Validation accuracy')
    axis[1].set_title('Training and Validation accuracy')
    axis[1].set_xlabel('Epoch')
    axis[1].set_ylabel('Accuracy Value')
    axis[1].set_ylim([0, 1])
    axis[1].legend()

    figure.tight_layout()
    if path_to_persist:
        figure.savefig(path_to_persist)
        plt.close(figure)
    else:
        print(f'Highest Validation Accuracy: {np.max(val_accuracy)}')
        plt.show()


def flatten_models(models, weight=1):
    individual_weight = weight / len(models)
    grouped_models = groupby(lambda m: isinstance(m, list), models)
    models = grouped_models.get(False, [])
    weighted_models = list(zip(models, [individual_weight] * len(models)))
    model_lists = grouped_models.get(True, [])
    for model in model_lists:
        weighted_models = weighted_models + flatten_models(model, weight=individual_weight)
    return weighted_models


def load_models(dataset_name=CurrentDatasets.swedish_leaf.value, model_names=['Encoder'], models_path=None):
    if models_path is None:
        models_path = f'../models/'
    models_path =  os.path.join(models_path, dataset_name)
    models_to_load = list(
        filter(lambda model_name: remove_suffix(model_name,'.h5') in model_names, os.listdir(models_path)))
    models = list(map(lambda filename: keras.models.load_model(models_path + "/" + filename), models_to_load))
    return models



def get_ensemble_predictions(x, models, evaluation_dataset, check_identical=False, models_path=None):
    """
    Create an ensemble of given models and dataset

    :param x: prediction input
    :param models: models to be ensembled
    :return: ensemble methods used and class predictions
    """
    models, weights = list(zip(*flatten_models(models)))
    models = load_models(evaluation_dataset, models, models_path=models_path)
    ensemble_methods = [method.value for method in EnsembleMethods]
    ensembles = list(map(lambda ensemble_type: Ensemble(models=models, weights=weights, ensemble_type=ensemble_type),
                         ensemble_methods))
    predictions = list(map(lambda ensemble: np.array(ensemble.__ensemble_method__(x, verbose=0)),
                           ensembles))
    # check for identical predictions
    if check_identical:
        identical_predictions_count = thread_last(combinations(predictions, 2),
                                                  (map, lambda comb: array_equal(comb[0], comb[1])),
                                                  (filter, identity),
                                                  count)
        print(f"{identical_predictions_count} identical predictions found")
    return ensemble_methods, predictions


def run_ensemble(evaluation_dataset=CurrentDatasets.swedish_leaf.value, 
                 model_names=['Encoder'], 
                 ensemble_name=None, 
                 models_path=None, 
                 augmentation=False, 
                 datasets_path='../datasets/'):
    """
    run an ensemble on a given dataset

    :param evaluation_dataset: dataset name
    :param model_names: model names to be ensembled, can be a nested list. Nested lists get weighted
    :param ensemble_name
    :return: list of accuracies and confusion matrices
    """
    if ensemble_name is None:
        ensemble_name = str(model_names)

    x_test, y_test = get_all_datasets_test_train_np_arrays(datasets_path, [evaluation_dataset])[evaluation_dataset][
        'test_data']
    if augmentation:
        x_test = np.array(list(map(add_additive_white_gaussian_noise, x_test)))

    method_names, predicted_classes = get_ensemble_predictions(x_test, model_names, evaluation_dataset, models_path=models_path)

    dataset_names = [evaluation_dataset for _ in method_names]
    display_names = [f"{ensemble_name}-{method_name}" for method_name in method_names]
    accuracies = list(map(partial(accuracy_score, y_test), predicted_classes))
    confusion_matrices = list(map(lambda predicted: json.dumps(confusion_matrix(y_test, predicted).tolist()),
                                  predicted_classes))
    return list(zip(dataset_names, display_names, accuracies, confusion_matrices))


def run_ensembles(dataset_names=[CurrentDatasets.swedish_leaf.value, CurrentDatasets.fifty_words.value],
                  ensembles={"All": ["Encoder", "FCN", "MCDCNN", "MLP", "Resnet", "Time_CNN"]},
                  verbose=False, 
                  models_path=None, 
                  datasets_path='../datasets/', 
                  augmentation=False):
    """
    run multiple ensembles on given datasets

    :param dataset_names
    :param ensembles: dictonary of ensembles
    :param verbose: prints progress
    :return: pandas dataframe with ensemble results
    """
    result = pandas.DataFrame(columns=['dataset_name', 'model_name', 'test_acc', 'confusion_matrix'])
    i = 1
    for dataset_name in dataset_names:
        if verbose:
            print(f"{i}/{len(dataset_names)}:\t{dataset_name}")
        for ensemble_name, model_names in ensembles.items():
            for row in run_ensemble(dataset_name, model_names, ensemble_name, models_path=models_path, datasets_path=datasets_path, augmentation=augmentation):
                result.loc[len(result)] = row
        i = i + 1
    return result
