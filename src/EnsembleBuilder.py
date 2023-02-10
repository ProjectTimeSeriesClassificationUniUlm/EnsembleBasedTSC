import os
import json
from itertools import combinations

from numpy import array_equal
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow import keras
from toolz import count, groupby, identity, thread_last

from tensorflow import keras
import pandas
from functools import partial
import numpy as np
from Ensemble import EnsembleMethods, Ensemble
from LoadData import CurrentDatasets, get_all_datasets_test_train_np_arrays
from PreprocessData import add_additive_white_gaussian_noise
from Helpers import remove_suffix


class EnsembleBuilder:
    def __init__(
        self,
        dataset_names=[
            CurrentDatasets.swedish_leaf.value,
            CurrentDatasets.fifty_words.value,
        ],
        ensembles={"All": ["Encoder-0", "FCN-0", "MCDCNN-0", "MLP-0", "Resnet-0", "Time_CNN-0"]},
        verbose=False,
        models_path=None,
        datasets_path="../datasets/",
        ensemble_methods=None,
    ):
        self.dataset_names = dataset_names
        self.verbose = verbose
        self.models_path = models_path
        self.datasets_path = datasets_path
        self.ensembles = ensembles
        if ensemble_methods is None:
            ensemble_methods = [method.value for method in EnsembleMethods]
        self.ensemble_methods = ensemble_methods

    def run_ensembles(self, augmentation=False):
        """
        run multiple ensembles on given datasets

        :param dataset_names
        :param ensembles: dictonary of ensembles
        :param verbose: prints progress
        :return: pandas dataframe with ensemble results
        """
        result = pandas.DataFrame(
            columns=["dataset_name", "model_name", "test_acc", "confusion_matrix"]
        )
        i = 1
        for dataset_name in self.dataset_names:
            if self.verbose:
                print(f"{i}/{len(self.dataset_names)}:\t{dataset_name}")
            for ensemble_name, model_names in self.ensembles.items():
                if self.verbose:
                    print(f"\t{ensemble_name}")
                for row in self._run_ensemble(
                    ensemble_name, model_names, dataset_name, augmentation
                ):
                    result.loc[len(result)] = row
            i = i + 1
        return result

    def _run_ensemble(
        self,
        ensemble_name,
        model_names=["Encoder"],
        evaluation_dataset=CurrentDatasets.swedish_leaf.value,
        augmentation=False,
    ):
        """
        run an ensemble on a given dataset

        :param evaluation_dataset: dataset name
        :param model_names: model names to be ensembled, can be a nested list. Nested lists get weighted
        :param ensemble_name
        :return: list of accuracies and confusion matrices
        """
        if ensemble_name is None:
            ensemble_name = str(self.model_names)
        x_test, y_test = get_all_datasets_test_train_np_arrays(
            self.datasets_path, [evaluation_dataset]
        )[evaluation_dataset]["test_data"]
        if augmentation:
            x_test = np.array(list(map(add_additive_white_gaussian_noise, x_test)))

        predicted_classes = self._get_ensemble_predictions(
            x_test, evaluation_dataset, model_names
        )

        dataset_names = [evaluation_dataset for _ in self.ensemble_methods]
        display_names = [
            f"{ensemble_name}-{method_name}" for method_name in self.ensemble_methods
        ]
        accuracies = list(map(partial(accuracy_score, y_test), predicted_classes))
        confusion_matrices = list(
            map(
                lambda predicted: json.dumps(
                    confusion_matrix(y_test, predicted).tolist()
                ),
                predicted_classes,
            )
        )
        return list(zip(dataset_names, display_names, accuracies, confusion_matrices))

    def _get_ensemble_predictions(
        self, x, evaluation_dataset, model_names, check_identical=False
    ):
        """
        Create an ensemble of given models and dataset

        :param x: prediction input
        :param models: models to be ensembled
        :return: ensemble methods used and class predictions
        """
        models, weights = list(zip(*self._flatten_models(model_names)))

        models = self._load_models(evaluation_dataset, models)
        ensembles = list(
            map(
                lambda ensemble_type: Ensemble(
                    models=models, weights=weights, ensemble_type=ensemble_type
                ),
                self.ensemble_methods,
            )
        )
        predictions = list(
            map(
                lambda ensemble: np.array(ensemble.__ensemble_method__(x, verbose=0)),
                ensembles,
            )
        )
        # check for identical predictions
        if check_identical:
            identical_predictions_count = thread_last(
                combinations(predictions, 2),
                (map, lambda comb: array_equal(comb[0], comb[1])),
                (filter, identity),
                count,
            )
            print(f"{identical_predictions_count} identical predictions found")
        return predictions

    def _load_models(
        self, dataset_name=CurrentDatasets.swedish_leaf.value, model_names=["Encoder"]
    ):
        if self.models_path is None:
            self.models_path = f"../models/"
        curr_models_path = os.path.join(self.models_path, dataset_name)
        models_to_load = list(
            filter(
                lambda model_name: remove_suffix(model_name, ".h5") in model_names,
                os.listdir(curr_models_path),
            )
        )
        models = list(
            map(
                lambda filename: keras.models.load_model(
                    curr_models_path + "/" + filename
                ),
                models_to_load,
            )
        )
        return models

    def _flatten_models(self, models, weight=1):
        individual_weight = weight / len(models)
        grouped_models = groupby(lambda m: isinstance(m, list), models)
        models = grouped_models.get(False, [])
        weighted_models = list(zip(models, [individual_weight] * len(models)))
        model_lists = grouped_models.get(True, [])
        for model in model_lists:
            weighted_models = weighted_models + self._flatten_models(
                model, weight=individual_weight
            )
        return weighted_models
