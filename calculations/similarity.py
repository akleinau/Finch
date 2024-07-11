import numpy as np
import pandas as pd
from calculations.item_functions import Item


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

    columns = get_columns(col_white_list, data, item_data)

    for col in columns:
        mean = data_std[col].mean()
        std = data_std[col].std()
        data_std[col] = (data_std[col] - mean) / std
        item_data[col] = (item_data[col] - mean) / std

    # calculate distance to item using euclidean distance
    data_std['distance'] = 0
    for col in columns:
        data_std['distance'] += (data_std[col] - item_data[col][0]) ** 2

    # get the 5% closest items, but at least 50 and all those that are very close
    data_std = data_std.sort_values(by='distance')
    num_items = min(max(int(len(data) * 0.05), 50), len(data_std))
    closest_density = data_std.head(num_items)
    min_distance = len(columns) * 0.1
    closest_distance = data_std[data_std['distance'] <= min_distance]
    combined_indexes = closest_density.index.union(closest_distance.index)

    # map back to original data
    data = data[data.index.isin(combined_indexes)]

    return data


def get_columns(col_white_list: list, data: pd.DataFrame, item_data: pd.DataFrame) -> list:
    """
    returns the columns that are used for the similarity calculation

    :param col_white_list: list
    :param data: pd.DataFrame
    :param item_data: pd.DataFrame
    :return: list
    """

    if len(col_white_list) == 0:
        columns = list(data.columns)
        excluded_columns = ['prob_', 'scatter', 'prediction', 'group', 'truth']

        columns = [col for col in columns if not any([excluded in col for excluded in excluded_columns])]
        item_columns = [col for col in item_data.columns if not any([excluded in col for excluded in excluded_columns])]
        columns = [col for col in columns if col in item_columns]
    else:
        columns = col_white_list
    return columns


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
    columns = get_columns(col_white_list, data, item_data)

    # replace each column with the item value
    for col in columns:
        data_pdp[col] = item_data[col].values[0]

    return data_pdp

def get_window_items(data, item, col, y_col):

    item_col_value = item.data_raw[col].values[0]
    mean_data = data.groupby(col).agg({y_col: 'mean'})

    window = get_window_size(mean_data)

    # get the items with an index that is within the window around the item
    item_index = mean_data.index.get_loc(item_col_value)
    start_index = max(0, item_index - window)
    end_index = item_index + 1
    mean_data = mean_data.iloc[start_index:end_index]

    return mean_data

def get_window_size(data):
    window = max(1, min(int(len(data) / 15), 10))
    window = window ** 2

    return window
