import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from toolz.functoolz import thread_first


def get_MLP(input_size, output_size):
    """
    Create a simple MLP model based on
    Z. Wang, W. Yan, and T. Oates, “Time series classification from scratch with deep neural networks: A strong baseline”, 2017
    :param input_size: number of features of the input data
    :param output_size: number of classes
    :return: keras sequential model of the MLP

    Is ResNet really better or just larger?
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


def get_Resnet(input_size, output_size, filters=64):
    """
    Create a Resnet Model based on
    Z. Wang, W. Yan, and T. Oates, “Time series classification from scratch with deep neural networks: A strong baseline”, 2017
    :param input_size: number of features of the input data
    :param output_size: number of classes
    :return: keras sequential model of the Encoder
    """
    inputs = keras.layers.Input((input_size, 1))

    convolutions1 = thread_first(
        inputs,
        keras.layers.Conv1D(filters=filters, kernel_size=8, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),

        keras.layers.Conv1D(filters=filters, kernel_size=5, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),

        keras.layers.Conv1D(filters=filters, kernel_size=3, padding='same'),
        keras.layers.BatchNormalization())
    shortcut1 = thread_first(
        inputs,
        keras.layers.Conv1D(filters=filters, kernel_size=1, padding='same'),
        keras.layers.BatchNormalization())
    block1 = thread_first(
        keras.layers.add([shortcut1, convolutions1]),
        keras.layers.Activation('relu'))

    convolutions2 = thread_first(
        block1,
        keras.layers.Conv1D(filters=filters * 2, kernel_size=8, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),

        keras.layers.Conv1D(filters=filters * 2, kernel_size=5, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),

        keras.layers.Conv1D(filters=filters * 2, kernel_size=3, padding='same'),
        keras.layers.BatchNormalization())
    shortcut2 = thread_first(
        block1,
        keras.layers.Conv1D(filters=filters * 2, kernel_size=1, padding='same'),
        keras.layers.BatchNormalization())
    block2 = thread_first(
        keras.layers.add([shortcut2, convolutions2]),
        keras.layers.Activation('relu'))

    convolutions3 = thread_first(
        block2,
        keras.layers.Conv1D(filters=filters * 2, kernel_size=8, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),

        keras.layers.Conv1D(filters=filters * 2, kernel_size=5, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),

        keras.layers.Conv1D(filters=filters * 2, kernel_size=3, padding='same'),
        keras.layers.BatchNormalization())
    shortcut3 = keras.layers.BatchNormalization()(block2)
    block3 = thread_first(
        keras.layers.add([shortcut3, convolutions3]),
        keras.layers.Activation('relu'),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(output_size, activation='softmax'))
    return keras.models.Model(inputs=inputs, outputs=block3)


def get_Encoder(input_size, output_size):
    """
    Create a Encoder Model based on
    Z. Wang, W. Yan, and T. Oates, “Time series classification from scratch with deep neural networks: A strong baseline”, 2017
    :param input_size: number of features of the input data
    :param output_size: number of classes
    :return: keras sequential model of the Encoder
    """
    inputs = keras.layers.Input((input_size, 1))
    convolutions = thread_first(
        inputs,
        keras.layers.Conv1D(filters=128, kernel_size=5, strides=1, padding='same'),
        tfa.layers.InstanceNormalization(),
        keras.layers.PReLU(shared_axes=[1]),
        keras.layers.Dropout(rate=0.2),
        keras.layers.MaxPooling1D(pool_size=2),

        keras.layers.Conv1D(filters=256, kernel_size=11, strides=1, padding='same'),
        tfa.layers.InstanceNormalization(),
        keras.layers.PReLU(shared_axes=[1]),
        keras.layers.Dropout(rate=0.2),
        keras.layers.MaxPooling1D(pool_size=2),

        keras.layers.Conv1D(filters=512, kernel_size=21, strides=1, padding='same'),
        tfa.layers.InstanceNormalization(),
        keras.layers.PReLU(shared_axes=[1]),
        keras.layers.Dropout(rate=0.2))
    attention_data = keras.layers.Lambda(lambda x: x[:, :, :256])(convolutions)
    attention_softmax = thread_first(
        convolutions,
        keras.layers.Lambda(lambda x: x[:, :, 256:]),
        keras.layers.Softmax())
    outputs = thread_first(
        [attention_data, attention_softmax],
        keras.layers.Multiply(),
        keras.layers.Dense(units=256, activation='sigmoid'),
        tfa.layers.InstanceNormalization(),
        keras.layers.Flatten(),
        keras.layers.Dense(units=output_size, activation='softmax'))
    return keras.models.Model(inputs=inputs, outputs=outputs)


def get_MCDCNN(input_size, output_size):
    return keras.Sequential([
        keras.layers.Input((input_size, 1)),

        keras.layers.Conv1D(filters=8, kernel_size=5, activation='sigmoid', input_shape=(input_size, 1), padding='same'),
        keras.layers.MaxPool1D(pool_size=2),

        keras.layers.Conv1D(filters=4, kernel_size=5, activation='sigmoid', padding='same'),
        keras.layers.MaxPool1D(pool_size=2),

        keras.layers.Flatten(),

        keras.layers.Dense(732, activation='sigmoid'),
        keras.layers.Dense(output_size, activation='softmax'),
    ])


def get_Time_CNN(input_size, output_size):
    return keras.Sequential([
        keras.layers.Conv1D(filters=6, kernel_size=7, activation='sigmoid', input_shape=(input_size, 1), padding='same'),
        keras.layers.AveragePooling1D(pool_size=3),

        keras.layers.Conv1D(filters=12, kernel_size=7, activation='sigmoid', padding='same'),
        keras.layers.AveragePooling1D(pool_size=3),

        keras.layers.Flatten(),

        keras.layers.Dense(output_size, activation='softmax'),
    ])
