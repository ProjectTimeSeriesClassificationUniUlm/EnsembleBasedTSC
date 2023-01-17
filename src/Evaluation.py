import os
import numpy as np
from matplotlib import pyplot as plt


def plot_model_history(history, epochs):
    loss = history['loss']
    val_loss = history['val_loss']

    accuracy = history['accuracy']
    val_accuracy = history['val_accuracy']
    print(f'Highest Validation Accuracy: {np.max(val_accuracy)}')
    eps = range(epochs)

    plt.figure()
    plt.plot(eps, loss, 'r', label='Training loss')
    plt.plot(eps, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 2])
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(eps, accuracy, 'r', label='Training accuracy')
    plt.plot(eps, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()

