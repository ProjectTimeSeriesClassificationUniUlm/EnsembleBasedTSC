from enum import Enum

import tensorflow as tf


class EnsembleMethods(str, Enum):
    AVERAGE = "average"
    LOGISTIC_AVERAGE = "logistic_average"
    MAJORITY_VOTE = "majority_vote"


class Ensemble(tf.keras.Model):
    """
    Ensemble of multiple keras models. Implements multiple methods for ensembling:
    - averaging
    - logistic averaging
    - majority vote
    """
    def __init__(self, models=None, ensemble_type: EnsembleMethods = EnsembleMethods.AVERAGE):
        super(Ensemble, self).__init__()
        if models is None:
            models = []
        self.models = models

        self.__ensemble_method__ = None
        match ensemble_type:
            case EnsembleMethods.AVERAGE:
                self.__ensemble_method__ = self.__average__
            case EnsembleMethods.LOGISTIC_AVERAGE:
                self.__ensemble_method__ = self.__logistic_average__
            case EnsembleMethods.MAJORITY_VOTE:
                self.__ensemble_method__ = self.__majority_vote__
            case _:
                raise ValueError("Invalid ensemble type")

    def call(self, x, *args, **kwargs):
        return self.__ensemble_method__(x)

    def get_all_predictions(self, x):
        return tf.stack([model(x)[0] for model in self.models])

    def __average__(self, x):
        pred = self.get_all_predictions(x)
        return tf.math.reduce_mean(pred, axis=0)

    def __logistic_average__(self, x):
        pred = self.get_all_predictions(x)
        return tf.math.reduce_mean(tf.math.sigmoid(pred), axis=0)

    def __majority_vote__(self, x):
        pred = self.get_all_predictions(x)
        argmax = tf.math.argmax(pred, axis=1)
        counts = tf.unique_with_counts(argmax)
        shape = tf.cast(pred[0].shape, dtype='int64')
        idx = tf.reshape(counts.y, (-1, 1))
        return tf.scatter_nd(idx, updates=counts.count, shape=shape)
