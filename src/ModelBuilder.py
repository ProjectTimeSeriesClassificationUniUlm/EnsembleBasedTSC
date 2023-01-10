import tensorflow as tf
from tensorflow import keras


def get_MLP(input_size, output_size):
    """
    Create a simple MLP model based on
    Z. Wang, W. Yan, and T. Oates, “Time series classification from scratch with deep neural networks: A strong baseline”, 2017
    :param input_size: number of features of the input data
    :param output_size: number of classes
    :return: keras sequential model of the MLP
    """
    return keras.Sequential([
        keras.layers.Dropout(0.1),
        keras.layers.Dense(500, activation='relu', input_shape=(input_size,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(500, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(500, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(output_size, activation='softmax'),
    ])


def get_FCN(input_size, output_size):
    """
    Create a Fully Convolutional Model based on
    Z. Wang, W. Yan, and T. Oates, “Time series classification from scratch with deep neural networks: A strong baseline”, 2017
    :param input_size: number of features of the input data
    :param output_size: number of classes
    :return: keras sequential model of the FCN
    """
    # How is the number of classes defined???
    return keras.Sequential([
        keras.layers.Conv1D(filters=128, kernel_size=8, input_shape=(input_size, 1)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),

        keras.layers.Conv1D(filters=256, kernel_size=5),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),

        keras.layers.Conv1D(filters=128, kernel_size=3),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),

        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(output_size, activation='softmax'),
    ])


def get_MCDCNN(input_size, output_size):
    return keras.Sequential([
        keras.layers.Conv1D(filters=8, kernel_size=5, activation='sigmoid', input_shape=(input_size, 1)),
        keras.layers.MaxPool1D(pool_size=2),

        keras.layers.Conv1D(filters=4, kernel_size=5, activation='sigmoid'),
        keras.layers.MaxPool1D(pool_size=2),

        keras.layers.Flatten(),

        keras.layers.Dense(732, activation='sigmoid'),
        keras.layers.Dense(output_size, activation='softmax'),
    ])


def get_Time_CNN(input_size, output_size):
    return keras.Sequential([
        keras.layers.Conv1D(filters=6, kernel_size=7, activation='sigmoid', input_shape=(input_size, 1)),
        keras.layers.AveragePooling1D(pool_size=3),

        keras.layers.Conv1D(filters=12, kernel_size=7, activation='sigmoid'),
        keras.layers.AveragePooling1D(pool_size=3),

        keras.layers.Flatten(),

        keras.layers.Dense(output_size, activation='softmax'),
    ])
