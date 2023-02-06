from enum import Enum

import tensorflow as tf
import numpy as np
from scipy.stats import stats
from sklearn.utils.extmath import weighted_mode
from toolz import thread_first, first


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

    def __init__(self, models=None, weights=None, ensemble_type: EnsembleMethods = EnsembleMethods.AVERAGE):
        super(Ensemble, self).__init__()
        if models is None:
            models = []
        if weights is None:
            weights = np.ones(len(models))
        # normalize weights
        weights = np.multiply(weights, 1.0 / sum(weights))
        self.models = models
        self.model_weights = weights
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

    def get_all_predictions(self, x, verbose="auto"):
        return tf.stack([model.predict(x, verbose=verbose) for model in self.models])

    def get_all_votes(self, x, verbose="auto"):
        return tf.stack([model.predict(x, verbose=verbose).argmax(axis=-1) for model in self.models])

    def __average__(self, x, verbose="auto"):
        pred = self.get_all_predictions(x, verbose)
        return tf.argmax(np.average(pred, axis=0, weights=self.model_weights), axis=-1)

    def __logistic_average__(self, x, verbose="auto"):
        pred = self.get_all_predictions(x, verbose)
        return tf.argmax(np.average(tf.math.sigmoid(pred), axis=0, weights=self.model_weights), axis=-1)

    def __majority_vote__(self, x, verbose="auto"):
        return thread_first(self.get_all_votes(x, verbose),
                            tf.stack,
                            (weighted_mode, np.transpose(np.array([self.model_weights]*len(x)))),  # axis is 0 by default
                            first,
                            first)
