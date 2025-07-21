import numpy as np
import pandas as pd

from calculations.item_functions import Item
from plots.helper_functions import check_if_categorical


def get_similar_items(data: pd.DataFrame, item: Item, col_white_list: list) -> pd.DataFrame:
    """
    depending on if pdp is used, starts the calculation of the similar items
    :param data: pd.DataFrame
    :param item: item_functions.Item
    :param col_white_list: list
    :return: pd.DataFrame
    """

    use_pdp = False
    if use_pdp:
        return get_pdp_items(data, item, col_white_list)
    else:
        return get_similar_subset(data, item, col_white_list)


def get_similar_subset(data: pd.DataFrame, item: Item, col_white_list: list) -> pd.DataFrame:
    """
    returns a subset of the data that is similar to the item

    :param data: pd.DataFrame
    :param item: item_functions.Item
    :param col_white_list: list
    :return: pd.DataFrame
    """

    # standardize the data
    data_std = data.copy()
    item_data = item.data_raw.copy()

    columns = col_white_list

    if len(columns) == 0:
        return pd.DataFrame()

    for col in columns:
        mean = data_std[col].mean()
        std = data_std[col].std()
        data_std[col] = (data_std[col] - mean) / std
        item_data[col] = (item_data[col] - mean) / std

    # select all items that are close enough in all columns
    close_boundary = 1 # threshold for closeness, can be adjusted
    for col in columns:
        data_std = data_std[data_std[col].between(item_data[col][0] - close_boundary, item_data[col][0] + close_boundary)]

    # map back to original data
    data = data[data.index.isin(data_std.index)]

    return data


def get_pdp_items(data, item, col_white_list):
    """
    INCOMPLETE FUNCTION - should return data adapted for PDP, but is not implemented yet and may not be necessary

    :param data:
    :param item:
    :param col_white_list:
    :return:
    """
    data_pdp = data.copy()
    item_data = item.data_raw.copy()
    columns = col_white_list

    # replace each column with the item value
    for col in columns:
        data_pdp[col] = item_data[col].values[0]

    return data_pdp


def get_window_items(data, item, col, y_col):
    item_col_value = item.data_raw[col].values[0]

    mean_data = data.groupby(col).agg({y_col: 'mean'})

    window = get_window_size(mean_data) - 1  # -1 because we add one to the end_index

    # get the closest item index to the item value
    # if the item value is not in the data, find the closest value
    item_index = np.abs(mean_data.index - item_col_value).argmin()

    # get the items with an index that is within the window around the item

    # categorical data with window center on the right
    if window == 1:
        start_index = max(0, item_index - window)
        end_index = item_index + 1
    # continuous data with window center on center
    else:
        start_index = max(0, item_index - window // 2)
        end_index = item_index + window // 2 + 1

    mean_data = mean_data.iloc[start_index:end_index]

    return mean_data


def get_window_size(data: pd.DataFrame, min_size=5) -> int:
    """
    calculates an appropriate window size based on the data size

    :param data: pd.DataFrame
    """

    if check_if_categorical(data):
        return 1

    window = min(max(min_size, int(len(data) * 0.05)), 1000) # min 5, max 1000, 0.05 of the data

    return window
