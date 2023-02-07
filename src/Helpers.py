import csv
import io
import json
from operator import concat

import keras
import numpy as np
import pandas as pd
from PyPDF2 import PdfMerger
from sklearn.metrics import confusion_matrix
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from toolz import groupby, valmap
from functools import partial, reduce


def append_to_csv(csv_path, input_row):
    with open(csv_path, 'a') as f_object:
        writer = csv.writer(f_object)
        writer.writerow(input_row)


def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string    


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def fold_nested(fn_leaf=lambda x: x, fn_node=sum, tree=[[1, 2], 3, [4, [5]]]):
    """
    (pre-order) Folds over a nested list

    :param fn_leaf: The function applied to leaf nodes/values
    :param fn_node: The function applied to nodes / result lists
    :param tree: the nested list to fold over
    :return: folded nested list
    """
    if isinstance(tree, list):  # node
        return fn_node(list(map(partial(fold_nested, fn_leaf, fn_node), tree)))
    else:  # leaf
        return fn_leaf(tree)


# flattens a list
flatten_list = partial(fold_nested, lambda x: [x], partial(reduce, concat))
