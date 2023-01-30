import csv
import io
import json

import keras
import numpy as np
import pandas as pd
from PyPDF2 import PdfMerger
from sklearn.metrics import confusion_matrix
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from toolz import groupby, valmap


def append_to_csv(csv_path, input_row):
    with open(csv_path, 'a') as f_object:
        writer = csv.writer(f_object)
        writer.writerow(input_row)


def get_confusion_matrix_for_model_and_data(model: keras.Model, x_test, y_test) -> np.ndarray:
    """
    Use the model to predict the classes of the test data and then return the confusion matrix. You can visualize them
    as a matplotlib plot using the visualize_confusion_matrix function.

    Usage:
    Returns a matrix C where C_{i, j} is equal to the number of observations known to be in group i but predicted to be in group j.
    That is if the model predicts the class 0 for 10 observations and the true class is 1 for 5 of them, then C[0, 1] = 5.

    For binary classification, C[0, 0] is the number of true negatives, C[0, 1] is the number of false positives,
    C[1, 0] is the number of false negatives, and C[1, 1] is the number of true positives. A perfect classifier
    would have C[0, 1] = C[1, 0] = 0. More general: In a perfect classifier, all entries of C are zero except for those
    on the main diagonal, which are equal to the number of observations in each group.

    :param model: The model to use for prediction.
    :param x_test: The test data.
    :param y_test: The test labels.
    :return: The confusion matrix as a numpy array.
    """
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    return confusion_matrix(y_test, y_pred)


def visualize_confusion_matrix(confusion_matrix_: np.ndarray, model_name: str, dataset_name: str) -> None:
    """
    Visualize the confusion matrix as a heatmap using seaborn and matplotlib.
    Model Name and Dataset Name are used for the title of the plot.

    :param confusion_matrix_: The confusion matrix to visualize generated by the get_confusion_matrix_for_model_and_data function.
    :param model_name: The name of the model.
    :param dataset_name: The name of the dataset.
    """
    plt.figure(figsize=(10, 10))
    sns.heatmap(confusion_matrix_, annot=True, fmt="d")
    plt.title(f"Confusion Matrix for {model_name} on {dataset_name}")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def create_confusion_matrix_plot_from_csv(csv_path: str, pdf_path=None, verbose=False):
    """
    Save the confusion matrices from given csv as a heatmap using seaborn and matplotlib as pdf.
    Model Name and Dataset Name are used for the title of the plot.

    :param csv_path: The csv file containing training data
    :param pdf_path: The pdf file name containing confusion matrices. Gets automatically generated from the csv_path
    :param verbose: plotting can take some time, prints current dataset name to stdout.
    """
    if pdf_path is None:
        pdf_path = csv_path.removesuffix('.csv') + '.pdf'
    results_dataframe = pd.read_csv(csv_path)
    confusion_matrices = results_dataframe[['dataset_name', 'model_name', 'confusion_matrix']].values.tolist()
    datasets = groupby(lambda x: x[0], confusion_matrices)  # create a dict with dataset as key, and matrices as values

    ncols = max(*valmap(len, datasets).values())
    # it is way faster to plot each dataset individually and then concat the resulting pdf
    # than letting matplotlib do everything at once
    merger = PdfMerger()
    row = 0
    for dataset_name, data in datasets.items():
        if verbose:
            print(dataset_name)
        fig, axs = plt.subplots(nrows=1, ncols=ncols, sharex=True, sharey=True, figsize=(ncols * 10, 10))
        col = 0
        for (_, model_name, confusion_matrix) in data:
            sns.heatmap(json.loads(confusion_matrix), annot=True, fmt="d", ax=axs[col])
            axs[col].set_title(f"{model_name} on {dataset_name}", fontsize=28)
            axs[col].set_ylabel('True label', fontsize=24)
            axs[col].set_xlabel('Predicted label', fontsize=24)
            col = col + 1
        pdf_buffer = io.BytesIO() # save intermediate pdf in-memory
        plt.savefig(pdf_buffer, format='pdf')
        plt.close()
        merger.append(pdf_buffer)
        merger.add_outline_item(dataset_name, row, None)
        pdf_buffer.close()
        row = row + 1
    merger.write(pdf_path)
    merger.close()


def get_project_root() -> Path:
    return Path(__file__).parent.parent
