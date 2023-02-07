from enum import Enum

import tensorflow as tf
import numpy as np
from sklearn.utils.extmath import weighted_mode
from toolz import thread_first, first


class EnsembleMethods(str, Enum):
    AVERAGE = "average"
    AVERAGE_WITH_CONFIDENCE = "average_with_confidence"
    LOGISTIC_AVERAGE = "logistic_average"
    LOGISTIC_AVERAGE_WITH_CONFIDENCE = "logistic_average_with_confidence"
    MAJORITY_VOTE = "majority_vote"


class Ensemble(tf.keras.Model):
    """
    Ensemble of multiple keras models. Implements multiple methods for ensembling:
    - averaging
    - logistic averaging
    - majority vote
    """

    def __init__(
        self,
        models=None,
        weights=None,
        ensemble_type: EnsembleMethods = EnsembleMethods.AVERAGE,
    ):
        super(Ensemble, self).__init__()
        if models is None:
            models = []
        if weights is None:
            weights = np.ones(len(models))
        # normalize weights
        weights = np.multiply(weights, 1.0 / sum(weights))
        self.models = models
        self.model_weights = weights
        ensemble_types = {
            EnsembleMethods.AVERAGE: self.__average__,
            EnsembleMethods.LOGISTIC_AVERAGE: self.__logistic_average__,
            EnsembleMethods.MAJORITY_VOTE: self.__majority_vote__,
            EnsembleMethods.AVERAGE_WITH_CONFIDENCE: self.__average_with_confidence__,
            EnsembleMethods.LOGISTIC_AVERAGE_WITH_CONFIDENCE: self.__logistic_average_with_confidence__,
        }
        self.__ensemble_method__ = ensemble_types.get(ensemble_type, None)
        if self.__ensemble_method__ is None:
            raise ValueError("Invalid ensemble type")

    def call(self, x, *args, **kwargs):
        return self.__ensemble_method__(x)

    def get_all_predictions(self, x, verbose="auto"):
        return tf.stack([model.predict(x, verbose=verbose) for model in self.models])

    def get_all_votes(self, x, verbose="auto"):
        return tf.stack(
            [model.predict(x, verbose=verbose).argmax(axis=-1) for model in self.models]
        )

    def __average__(self, x, verbose="auto"):
        pred = self.get_all_predictions(x, verbose)
        return tf.argmax(np.average(pred, axis=0, weights=self.model_weights), axis=-1)

    def __logistic_average__(self, x, verbose="auto"):
        pred = self.get_all_predictions(x, verbose)
        return tf.argmax(
            np.average(tf.math.sigmoid(pred), axis=0, weights=self.model_weights),
            axis=-1,
        )

    def __majority_vote__(self, x, verbose="auto"):
        return thread_first(
            self.get_all_votes(x, verbose),
            tf.stack,
            (
                weighted_mode,
                np.transpose(np.array([self.model_weights] * len(x))),
            ),  # axis is 0 by default
            first,
            first,
        )

    def __average_with_confidence__(self, x, verbose="auto"):
        pred = self.get_all_predictions(x, verbose)
        return tf.argmax(
            np.apply_along_axis(self.__calculate_column_avg_with_confidence__, 0, pred),
            axis=-1,
        )

    def __logistic_average_with_confidence__(self, x, verbose="auto"):
        pred = self.get_all_predictions(x, verbose)
        return tf.argmax(
            np.apply_along_axis(
                self.__calculate_column_avg_with_confidence__, 0, tf.math.sigmoid(pred)
            ),
            axis=-1,
        )

    def __calculate_column_avg_with_confidence__(self, pred_column):
        return np.average(
            pred_column, axis=0, weights=self.model_weights * pred_column * 10
        )
