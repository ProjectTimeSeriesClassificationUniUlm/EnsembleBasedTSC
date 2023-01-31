import json
import os
from functools import partial

import numpy as np
import tensorflow_addons # needed for model import, do not remove
import pandas
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow import keras
from Ensemble import EnsembleMethods, Ensemble
from LoadData import CurrentDatasets, get_all_datasets_test_train_np_arrays


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


def get_ensemble_predictions(x, models):
    """
    Create an ensemble of given models and dataset

    :param x: prediction input
    :param models: models to be ensembled
    :return: ensemble methods used and class predictions
    """
    ensemble_methods = [method.value for method in EnsembleMethods]
    ensembles = list(map(lambda ensemble_type: Ensemble(models=models, ensemble_type=ensemble_type),
                         ensemble_methods))
    return ensemble_methods, list(map(lambda ensemble: np.array(ensemble.__ensemble_method__(x)),
                                      ensembles))


def run_ensemble(evaluation_dataset=CurrentDatasets.swedish_leaf.value, model_names=['Encoder'], ensemble_name=None):
    """
    run an ensemble on a given dataset

    :param evaluation_dataset: dataset name
    :param model_names: model names to be ensembled
    :param ensemble_name
    :return: list of accuracies and confusion matrices
    """
    if ensemble_name is None:
        ensemble_name = str(model_names)

    x_test, y_test = get_all_datasets_test_train_np_arrays('../datasets/', [evaluation_dataset])[evaluation_dataset][
        'test_data']
    model_path = f'../models/{evaluation_dataset}'
    models_to_load = list(
        filter(lambda model_name: model_name.removesuffix('.h5') in model_names, os.listdir(model_path)))
    models = list(map(lambda filename: keras.models.load_model(model_path + "/" + filename), models_to_load))

    method_names, predicted_classes = get_ensemble_predictions(x_test, models)

    dataset_names = [evaluation_dataset for _ in method_names]
    display_names = [f"{ensemble_name}-{method_name}" for method_name in method_names]
    accuracies = list(map(partial(accuracy_score, y_test), predicted_classes))
    confusion_matrices = list(map(lambda predicted: json.dumps(confusion_matrix(y_test, predicted).tolist()),
                                  predicted_classes))
    return list(zip(dataset_names, display_names, accuracies, confusion_matrices))


def run_ensembles(dataset_names=[CurrentDatasets.swedish_leaf.value, CurrentDatasets.fifty_words.value],
                  ensembles={"All": ["Encoder", "FCN", "MCDCNN", "MLP", "Resnet", "Time_CNN"]},
                  verbose=False):
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
            for row in run_ensemble(dataset_name, model_names, ensemble_name):
                result.loc[len(result)] = row
        i = i + 1
    return result
