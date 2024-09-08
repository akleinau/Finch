import numpy as np


def check_if_categorical(data, col=None):
    if col is None:
        # first extract index
        unique_values = list(data.index)
        return len(unique_values) <= 24
    else:
        unique_values = data[col].unique()
        return len(unique_values) <= 24