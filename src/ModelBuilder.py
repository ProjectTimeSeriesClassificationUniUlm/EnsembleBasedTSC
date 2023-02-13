import tensorflow as tf
import tensorflow_addons as tfa
import json
from tensorflow import keras
from toolz.functoolz import thread_first


def get_model_name(model_builder):
    return model_builder.__name__.replace("get_", "")


def get_MLP(input_size, output_size):
    """
    Create a simple MLP model based on
    Z. Wang, W. Yan, and T. Oates, “Time series classification from scratch with deep neural networks: A strong baseline”, 2017
    :param input_size: number of features of the input data
    :param output_size: number of classes
    :return: keras sequential model of the MLP

    Is ResNet really better or just larger?
    """
    return keras.Sequential(
        [
            keras.layers.Dropout(0.1),
            keras.layers.Dense(
                500,
                activation="relu",
                input_shape=(input_size,),
                kernel_initializer=tf.keras.initializers.GlorotUniform(),
            ),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(
                500,
                activation="relu",
                kernel_initializer=tf.keras.initializers.GlorotUniform(),
            ),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(
                500,
                activation="relu",
                kernel_initializer=tf.keras.initializers.GlorotUniform(),
            ),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(
                output_size,
                activation="softmax",
                kernel_initializer=tf.keras.initializers.GlorotUniform(),
            ),
        ]
    )


def get_FCN(input_size, output_size, transfer_learning=False):
    """
    Create a Fully Convolutional Model based on
    Z. Wang, W. Yan, and T. Oates, “Time series classification from scratch with deep neural networks: A strong baseline”, 2017
    :param input_size: number of features of the input data
    :param output_size: number of classes
    :return: keras sequential model of the FCN
    """
    first_layer = keras.layers.Conv1D(filters=256, kernel_size=8, padding="same", kernel_initializer=tf.keras.initializers.GlorotUniform()) \
        if transfer_learning \
        else keras.layers.Conv1D(filters=256, kernel_size=8, padding="same", input_shape=(input_size, 1), kernel_initializer=tf.keras.initializers.GlorotUniform())

    return keras.Sequential(
        [
            first_layer,
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.Conv1D(
                filters=256,
                kernel_size=5,
                padding="same",
                kernel_initializer=tf.keras.initializers.GlorotUniform(),
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.Conv1D(
                filters=128,
                kernel_size=3,
                padding="same",
                kernel_initializer=tf.keras.initializers.GlorotUniform(),
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(
                output_size,
                activation="softmax",
                kernel_initializer=tf.keras.initializers.GlorotUniform(),
            ),
        ]
    )


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
        keras.layers.Conv1D(
            filters=filters,
            kernel_size=8,
            padding="same",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        ),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Conv1D(
            filters=filters,
            kernel_size=5,
            padding="same",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        ),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Conv1D(
            filters=filters,
            kernel_size=3,
            padding="same",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        ),
        keras.layers.BatchNormalization(),
    )
    shortcut1 = thread_first(
        inputs,
        keras.layers.Conv1D(
            filters=filters,
            kernel_size=1,
            padding="same",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        ),
        keras.layers.BatchNormalization(),
    )
    block1 = thread_first(
        keras.layers.add([shortcut1, convolutions1]), keras.layers.Activation("relu")
    )

    convolutions2 = thread_first(
        block1,
        keras.layers.Conv1D(
            filters=filters * 2,
            kernel_size=8,
            padding="same",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        ),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Conv1D(
            filters=filters * 2,
            kernel_size=5,
            padding="same",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        ),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Conv1D(
            filters=filters * 2,
            kernel_size=3,
            padding="same",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        ),
        keras.layers.BatchNormalization(),
    )
    shortcut2 = thread_first(
        block1,
        keras.layers.Conv1D(
            filters=filters * 2,
            kernel_size=1,
            padding="same",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        ),
        keras.layers.BatchNormalization(),
    )
    block2 = thread_first(
        keras.layers.add([shortcut2, convolutions2]), keras.layers.Activation("relu")
    )

    convolutions3 = thread_first(
        block2,
        keras.layers.Conv1D(
            filters=filters * 2,
            kernel_size=8,
            padding="same",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        ),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Conv1D(
            filters=filters * 2,
            kernel_size=5,
            padding="same",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        ),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Conv1D(
            filters=filters * 2,
            kernel_size=3,
            padding="same",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        ),
        keras.layers.BatchNormalization(),
    )
    shortcut3 = keras.layers.BatchNormalization()(block2)
    block3 = thread_first(
        keras.layers.add([shortcut3, convolutions3]),
        keras.layers.Activation("relu"),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(output_size, activation="softmax"),
    )
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
        keras.layers.Conv1D(
            filters=128,
            kernel_size=5,
            strides=1,
            padding="same",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        ),
        tfa.layers.InstanceNormalization(),
        keras.layers.PReLU(shared_axes=[1]),
        keras.layers.Dropout(rate=0.2),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Conv1D(
            filters=256,
            kernel_size=11,
            strides=1,
            padding="same",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        ),
        tfa.layers.InstanceNormalization(),
        keras.layers.PReLU(shared_axes=[1]),
        keras.layers.Dropout(rate=0.2),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Conv1D(
            filters=512,
            kernel_size=21,
            strides=1,
            padding="same",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        ),
        tfa.layers.InstanceNormalization(),
        keras.layers.PReLU(shared_axes=[1]),
        keras.layers.Dropout(rate=0.2),
    )
    attention_data = keras.layers.Lambda(lambda x: x[:, :, :256])(convolutions)
    attention_softmax = thread_first(
        convolutions,
        keras.layers.Lambda(lambda x: x[:, :, 256:]),
        keras.layers.Softmax(),
    )
    outputs = thread_first(
        [attention_data, attention_softmax],
        keras.layers.Multiply(),
        keras.layers.Dense(
            units=256,
            activation="sigmoid",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        ),
        tfa.layers.InstanceNormalization(),
        keras.layers.Flatten(),
        keras.layers.Dense(
            units=output_size,
            activation="softmax",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        ),
    )
    return keras.models.Model(inputs=inputs, outputs=outputs)


def get_MCDCNN(input_size, output_size):
    return keras.Sequential(
        [
            keras.layers.Input((input_size, 1)),
            keras.layers.Conv1D(
                filters=8,
                kernel_size=5,
                activation="sigmoid",
                input_shape=(input_size, 1),
                padding="same",
                kernel_initializer=tf.keras.initializers.GlorotUniform(),
            ),
            keras.layers.MaxPool1D(pool_size=2),
            keras.layers.Conv1D(
                filters=4, kernel_size=5, activation="sigmoid", padding="same"
            ),
            keras.layers.MaxPool1D(pool_size=2),
            keras.layers.Flatten(),
            keras.layers.Dense(
                732,
                activation="sigmoid",
                kernel_initializer=tf.keras.initializers.GlorotUniform(),
            ),
            keras.layers.Dense(
                output_size,
                activation="softmax",
                kernel_initializer=tf.keras.initializers.GlorotUniform(),
            ),
        ]
    )


def get_Time_CNN(input_size, output_size):
    return keras.Sequential(
        [
            keras.layers.Conv1D(
                filters=6,
                kernel_size=7,
                activation="sigmoid",
                input_shape=(input_size, 1),
                padding="same",
                kernel_initializer=tf.keras.initializers.GlorotUniform(),
            ),
            keras.layers.AveragePooling1D(pool_size=3),
            keras.layers.Conv1D(
                filters=12,
                kernel_size=7,
                activation="sigmoid",
                padding="same",
                kernel_initializer=tf.keras.initializers.GlorotUniform(),
            ),
            keras.layers.AveragePooling1D(pool_size=3),
            keras.layers.Flatten(),
            keras.layers.Dense(
                output_size,
                activation="softmax",
                kernel_initializer=tf.keras.initializers.GlorotUniform(),
            ),
        ]
    )

def get_MCDCNN_improved(input_size, output_size):
    return build_MCDCNN(input_size, output_size, **json.loads("""{"nr_conv_layers": 2, "nr_conv_filters": 9, "kernel_size": 
    7, "padding_method": "same", "dense_size": 196, "nr_dense_layers": 2, "conv_activation": "relu", 
    "dense_activation": "relu", "pooling_method": "max_pool", "pooling_size": 2, "dropout_rate": 0.1, 
    "use_batchnorm": true}"""))

def build_MCDCNN(input_size, output_size,
                 conv_config=(8, 4), kernel_size=5, padding_method="same", dense_config=(732, ),
                 conv_activation="sigmoid", dense_activation="sigmoid",
                 pooling_method=keras.layers.MaxPool1D, pooling_size=2,
                 dropout_rate=0.5, use_batchnorm=False):
    """
    Build a MCDCNN model which architecture is defined by the parameters.
    :param input_size: Size of the input
    :param output_size: Size of the output
    :param conv_config: List of filters for each convolutional layer -> [filters, ...]
    :param kernel_size: Size of the kernel for the convolutional layers
    :param padding_method: The method used for padding the input (do not use "valid" padding) -> "same"  | "causal"
    :param dense_config: List of units for each dense layer -> [units, ...]
    :param conv_activation: Activation function for the convolutional layers
    :param dense_activation: Activation function for the dense layers
    :param pooling_method: The method used for pooling the output of the convolutional layers -> "avg_pool" | "max_pool"
    :param pooling_size: The size of the pooling window
    :param dropout_rate: The rate of the dropout layers. If dropout_rate=0, no dropout layers will be added
    :param use_batchnorm: If True, batch normalization layers will be added after each convolutional and dense layer
    """
    if dropout_rate >= 1:
        raise ValueError(f"Dropout rate must be between 0 and 1 but is set to {dropout_rate}")
    if not conv_config or len(conv_config) == 0:
        raise ValueError(f"Convolutional configuration must not be empty but is set to {conv_config}")
    if not dense_config or len(dense_config) == 0:
        raise ValueError(f"Dense configuration must not be empty but is set to {dense_config}")

    model = keras.Sequential()
    model.add(keras.layers.Input((input_size, 1)))

    # Convolutional Layers
    for filters in conv_config:
        model.add(keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            activation=conv_activation,
            input_shape=(input_size, 1),
            padding=padding_method,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        ))

        if use_batchnorm:
            model.add(keras.layers.BatchNormalization())
        if dropout_rate > 0:
            model.add(keras.layers.Dropout(dropout_rate))

        model.add(
            keras.layers.MaxPooling1D(pool_size=pooling_size) if pooling_method == "max_pool"
            else keras.layers.AveragePooling1D(pool_size=pooling_size)
        )


    # Flatten Layer to feed Dense Layers
    model.add(keras.layers.Flatten())

    # Dense Layers
    for units in dense_config:
        model.add(keras.layers.Dense(
            units,
            activation=dense_activation,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        ))
        if use_batchnorm:
            model.add(keras.layers.BatchNormalization())
        if dropout_rate > 0:
            model.add(keras.layers.Dropout(dropout_rate))

    # Output Layer
    model.add(keras.layers.Dense(
        output_size,
        activation="softmax",
        kernel_initializer=tf.keras.initializers.GlorotUniform(),
    ))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
