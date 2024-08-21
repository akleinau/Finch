import numpy as np
import pandas as pd
from calculations.similarity import get_window_items, get_similar_items
from panel.viewable import Viewer
import param


class Recommendation(Viewer):
    single_dict = param.ClassSelector(class_=dict)
    dataset_single = param.ClassSelector(class_=pd.DataFrame)
    dataset_overview = param.ClassSelector(class_=pd.DataFrame)

    def __init__(self, **params):
        super().__init__(**params)
        self.single_dict = None
        self.dataset_single = None
        self.dataset_overview = None
        self.mean_prob = None
        self.remaining_columns = None
        self.data = None
        self.item = None
        self.y_col = None
        self.columns = None

    def update_item(self, data, item, y_col, columns, all_selected_cols):
        self.data = data
        self.item = item
        self.y_col = y_col
        self.columns = columns

        self.remaining_columns = [col for col in columns if col not in all_selected_cols]
        self.mean_prob = data[y_col].mean()

        single_dict = {}
        for col in columns:
            # get prediction of the col on its own
            single_dict[col] = get_window_items(data, item, col, y_col)[y_col].mean() - self.mean_prob
        self.single_dict = single_dict

        self.dataset_single = get_dataset(data, item, y_col, self.remaining_columns, all_selected_cols, single_dict,
                                          self.mean_prob)

        self.dataset_overview = get_overview_dataset(data, item, y_col, columns, single_dict, self.mean_prob)

    def update_selected_cols(self, all_selected_cols):
        self.remaining_columns = [col for col in self.columns if col not in all_selected_cols]
        self.dataset_single = get_dataset(self.data, self.item, self.y_col, self.remaining_columns, all_selected_cols,
                                          self.single_dict, self.mean_prob)


def get_dataset(data, item, y_col, remaining_columns, all_selected_cols, single_dict, mean_prob):
    """
    creates the dataset for the tornado plot, containing all single features and the interaction strength
    """

    # get all singular features
    results = []
    for col in remaining_columns:
        columns = all_selected_cols + [col]

        single_value = single_dict[col]

        if len(columns) > 1:
            first_col = all_selected_cols[0]

            # calculate added value
            if len(columns) > 2:
                similar_items = get_similar_items(data, item, all_selected_cols[1:] + [col])
            else:
                similar_items = data
            prev_value = get_window_items(similar_items, item, first_col, y_col)[y_col].mean() - mean_prob
            added_value = single_value + prev_value

            # calculate joined value
            # get prediction of the col with the other selected cols
            similar_items = get_similar_items(data, item, columns[1:])
            joined_value = get_window_items(similar_items, item, first_col, y_col)[y_col].mean() - mean_prob

            # value = single_mean - mean_prob
            value = np.abs(joined_value - added_value)

            results.append({'feature': col, 'prediction': joined_value, 'value': value,
                            'item_value': item.data_raw[col].values[0]})
        else:
            results.append({'feature': col, 'prediction': single_value, 'value': single_value,
                            'item_value': item.data_raw[col].values[0]})

    dataframe = create_dataframe(results)

    return dataframe


def get_overview_dataset(data, item, y_col, columns, single_dict, mean_prob):
    """
    creates the dataset for the overview plot, containing the most important feature interactions
    """

    # if the data set size is too big, only take a sample
    if len(data) > 1000:
        data = data.copy().sample(1000)

    prob_range = 0.5 * data[y_col].std()
    min_value = prob_range

    # recursively build a tree for each column
    tree_list = []
    for col in columns:
        remaining = [c for c in columns if c != col]
        tree_list.append(InteractTreeRoot(single_dict, col, remaining, mean_prob, data, item, y_col, min_value))

    # create a list of all nodes
    nodes = []
    for tree in tree_list:
        nodes.extend(tree.get_nodes())

    # create a dataframe from the nodes
    results = []
    for node in nodes:
        results.append({'feature': ", ".join(node.columns), 'value': node.value, 'prediction': node.prediction,
                        'added': node.added, 'prev_value': node.prev_value})

    dataframe = create_dataframe(results)

    return dataframe


def create_dataframe(results):
    """
    creates a dataframe with some additional info from the results
    """

    dataframe = pd.DataFrame(results)
    if len(dataframe) == 0:
        return dataframe

    dataframe['abs_value'] = dataframe['value'].abs()
    dataframe['positive'] = dataframe['value'].apply(lambda x: 'pos' if x > 0 else 'neg')
    dataframe = dataframe.sort_values(by='abs_value', ascending=True)
    dataframe.reset_index(drop=True, inplace=True)

    if "item_value" in dataframe.columns:
        dataframe['feature_name'] = dataframe['feature'] + " = " + dataframe['item_value'].apply(lambda x: str(round(x, 2)))

    return dataframe


class InteractTree:
    value = 0
    prediction = 0
    added = 0
    prev_value = 0
    columns = []
    count = 0


class InteractTreeRoot(InteractTree):

    def __init__(self, single_dict, col, remaining_columns, mean_prob, data, item, y_col, min_value):
        self.value = np.abs(single_dict[col])
        self.prediction = single_dict[col]
        self.added = single_dict[col]
        self.prev_value = 0
        self.count = 1
        self.columns = [col]
        self.nodes = []
        if self.value > min_value:
            for col in remaining_columns:
                remaining = [c for c in remaining_columns if c != col]
                self.nodes.append(
                    InteractTreeSub(single_dict, self, col, remaining, mean_prob, data, item, y_col, min_value))

    def get_nodes(self):
        nodes = []
        for node in self.nodes:
            nodes.extend(node.get_nodes())
        return nodes


class InteractTreeSub(InteractTree):
    def __init__(self, single_dict, prev, col, remaining_columns, mean_prob, data, item, y_col, min_value):
        # calculate added value
        single_value = single_dict[col]
        prev_value = prev.prediction
        added_value = single_value + prev_value

        # calculate joined value
        # get prediction of the col with the other selected cols
        first_col = prev.columns[0]
        similar_items = get_similar_items(data, item, [col] + prev.columns[1:])
        joined_value = get_window_items(similar_items, item, first_col, y_col)[y_col].mean() - mean_prob

        # set all values
        self.prediction = joined_value
        self.added = added_value
        self.prev_value = prev_value
        self.columns = prev.columns + [col]
        self.count = len(self.columns)
        difference = np.abs(joined_value - added_value)
        self.value = (prev.value * prev.count + difference) / self.count
        # self.raw_value = np.abs(joined_value - added_value)
        # self.value = 0 if min_value/2 >= self.raw_value else (prev.value * prev.count + np.abs(joined_value - added_value)) / self.count

        max_depth = 3 if len(prev.columns) + len(remaining_columns) < 15 else 2

        self.nodes = []
        if len(self.columns) and difference > min_value and self.count < max_depth:
            for col in remaining_columns:
                remaining = [c for c in remaining_columns if c != col]
                self.nodes.append(
                    InteractTreeSub(single_dict, self, col, remaining, mean_prob, data, item, y_col, min_value))

    def get_nodes(self):
        nodes = [self]
        for node in self.nodes:
            nodes.extend(node.get_nodes())
        return nodes
