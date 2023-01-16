import numpy as np
from sktime.datasets import load_from_tsfile_to_dataframe

def _prepare_train_data(df):
    prepared = list()
    for j in range(df.size):
        prepared.append(df.iloc[j].to_numpy()[0].to_numpy())
    return np.array(prepared, dtype=np.float64)

def load_numpy_array_from_ts(path_to_file):
    x, y = load_from_tsfile_to_dataframe(path_to_file)
    return _prepare_train_data(x), np.array(y, dtype=np.float64)