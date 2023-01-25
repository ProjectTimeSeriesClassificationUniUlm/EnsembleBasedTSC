import os
import numpy as np
from matplotlib import pyplot as plt


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

