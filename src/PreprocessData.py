from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow import keras
import random
import tensorflow as tf
import numpy as np
from tensorflow.data import AUTOTUNE


def preprocess_datasets(datasets, augmentation=False, batch_size=10):
    for ds_name, ds_data in datasets.items():
        ds_data["input_size"] = ds_data["train_data"][0].shape[1]
        ds_data["output_size"] = len(np.unique(ds_data["train_data"][1]))
        ds_data["train_data"] = preprocess_data(
            ds_data["train_data"][0],
            ds_data["train_data"][1],
            augmentation=augmentation,
            batch_size=batch_size,
        )
    return datasets


def preprocess_data(x_data, y_data, augmentation=False, batch_size=10):
    data_augmentation = keras.Sequential(
        [
            AdditiveWhiteGaussianNoise(),
        ]
    )
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    if augmentation:
        # dataset = dataset
        dataset = (
            dataset.map(lambda x, y: (data_augmentation(x), y))
            .batch(batch_size)
            .prefetch(buffer_size=AUTOTUNE)
        )
    else:
        dataset = dataset.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    return dataset


class AdditiveWhiteGaussianNoise(Layer):
    """
    Add additive white gaussian noise to the dataset.
    The noise standart deviation is a random value from the defined interval [min_noise_stddev, max_noise_stddev],
    that is randomly taken during each layer's call.
    Input shape:
      One dimensional array with time series data. Number of points is arbitrary.
    Output shape:
      Same as input.
    Arguments:
      min_noise_stddev: Int, the scale to apply to the inputs.
      max_noise_stddev: Int, the offset to apply to the inputs.
      name: A string, the name of the layer.
    """

    def __init__(
        self,
        min_noise_stddev: int = 0.05,
        max_noise_stddev: int = 0.4,
        name="AdditiveWhiteGaussianNoise",
        **kwargs
    ):
        self.min_noise_stddev = min_noise_stddev
        self.max_noise_stddev = max_noise_stddev
        super(AdditiveWhiteGaussianNoise, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        noise_stddev = random.uniform(self.min_noise_stddev, self.max_noise_stddev)
        outputs = tf.py_function(
            add_additive_white_gaussian_noise, [inputs, noise_stddev], np.float32
        )
        outputs.set_shape(inputs.shape)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "min_noise_stddev": self.min_noise_stddev,
            "max_noise_stddev": self.max_noise_stddev,
        }
        base_config = super(AdditiveWhiteGaussianNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def add_additive_white_gaussian_noise(ts, noise_stddev=0.2):
    # generate random numbers from a Gaussian distribution with the given standard deviation
    noise = np.random.normal(0, noise_stddev, len(ts))
    # add the noise to the signal
    noisy_signal = ts + noise
    return noisy_signal
