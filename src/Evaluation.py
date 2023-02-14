import json

import io
import numpy as np
import pandas as pd
import tensorflow_addons  # needed for model import, do not remove
import pandas
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from toolz import groupby
from PyPDF2 import PdfMerger
import matplotlib.pyplot as plt
import seaborn as sns

from Helpers import remove_suffix


def plot_model_history(history, epochs=None, path_to_persist=None):
    if epochs is None:
        print("No epochs specified, using all epochs.")
        epochs = len(history["loss"])

    else:
        print(
            f"Using {epochs} epochs. You can ignore the epochs parameter if you want to use all epochs."
        )

    loss = history["loss"]
    val_loss = history["val_loss"]

    accuracy = history["accuracy"]
    val_accuracy = history["val_accuracy"]
    eps = range(epochs)

    figure, axis = plt.subplots(2, 1)
    axis[0].plot(eps, loss, "r", label="Training loss")
    axis[0].plot(eps, val_loss, "b", label="Validation loss")
    axis[0].set_title("Training and Validation Loss")
    axis[0].set_xlabel("Epoch")
    axis[0].set_ylabel("Loss Value")
    axis[0].set_ylim([0, 2])
    axis[0].legend()

    axis[1].plot(eps, accuracy, "r", label="Training accuracy")
    axis[1].plot(eps, val_accuracy, "b", label="Validation accuracy")
    axis[1].set_title("Training and Validation accuracy")
    axis[1].set_xlabel("Epoch")
    axis[1].set_ylabel("Accuracy Value")
    axis[1].set_ylim([0, 1])
    axis[1].legend()

    figure.tight_layout()
    if path_to_persist:
        figure.savefig(path_to_persist)
        plt.close(figure)
    else:
        print(f"Highest Validation Accuracy: {np.max(val_accuracy)}")
        plt.show()


def plot_accuracies_scatter_plot(
    df,
    title="Test average accuracies",
    x_column="avg_test_acc",
    y_column="model_name",
    path_to_persist=None,
):
    sns.set(
        rc={
            "grid.color": "grey",
            "grid.linestyle": ":",
            "figure.figsize": (10, 6),
            "axes.labelsize": 15,
            "axes.titlesize": 20,
        }
    )
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, s=100, x=x_column, y=y_column).set(title=title)
    if path_to_persist:
        fig.savefig(path_to_persist)
        plt.close(figure)


def plot_accuracies_bar_plot(
    df,
    title="Test average accuracies",
    x_column="avg_test_acc",
    y_column="model_name",
    path_to_persist=None,
    figsize = (10,15)
):
    sns.set(rc={"grid.color": "grey",
                "grid.linestyle": ":",
                "figure.figsize":(10,6),
                "axes.labelsize":15,
                "axes.titlesize":20})
    plt.figure(figsize = figsize)
    plt.xlim(0, 1)
    splot = sns.barplot(data=df, x=x_column, y=y_column, palette=sns.color_palette("Spectral", len(df)))
    splot.set(title=title)
    show_values_on_bar_plot(splot, "h", space=0)
    if path_to_persist:
        fig.savefig(path_to_persist)
        plt.close(figure)


def show_values_on_bar_plot(axs, orient="v", space=.01):
    def _single(ax):
        if orient == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height()*0.01)
                value = '{:.3f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center") 
        elif orient == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*0.5)
                value = '{:.3f}'.format(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)
        

def get_confusion_matrix_for_model_and_data(
    model: keras.Model, x_test, y_test
) -> np.ndarray:
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


def visualize_confusion_matrix(
    confusion_matrix_: np.ndarray, model_name: str, dataset_name: str
) -> None:
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
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
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
        pdf_path = remove_suffix(csv_path, ".csv") + ".pdf"
    results_dataframe = pd.read_csv(csv_path)
    confusion_matrices = results_dataframe[
        ["dataset_name", "model_name", "confusion_matrix", "test_acc"]
    ].values.tolist()
    datasets = groupby(
        lambda x: x[0], confusion_matrices
    )  # create a dict with dataset as key, and matrices as values

    # it is way faster to plot each dataset individually and then concat the resulting pdf
    # than letting matplotlib do everything at once
    merger = PdfMerger()
    row = 0
    for dataset_name, data in datasets.items():
        ncols = len(data)
        if verbose:
            print(dataset_name)
        fig, axs = plt.subplots(
            nrows=1, ncols=ncols, sharex=True, sharey=True, figsize=(ncols * 10, 10)
        )
        col = 0
        for _, model_name, confusion_matrix, test_acc in data:
            sns.heatmap(json.loads(confusion_matrix), annot=True, fmt="d", ax=axs[col])
            axs[col].set_title(
                f"{model_name} on {dataset_name}\naccuracy: {round(test_acc, 3)}",
                fontsize=28,
            )
            axs[col].set_ylabel("True label", fontsize=24)
            axs[col].set_xlabel("Predicted label", fontsize=24)
            col = col + 1
        pdf_buffer = io.BytesIO()  # save intermediate pdf in-memory
        plt.savefig(pdf_buffer, format="pdf")
        plt.close()
        merger.append(pdf_buffer)
        merger.add_outline_item(dataset_name, row, None)
        pdf_buffer.close()
        row = row + 1
    merger.write(pdf_path)
    merger.close()
