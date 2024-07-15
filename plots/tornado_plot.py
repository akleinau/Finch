import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.models import ColumnDataSource, HoverTool
import param

from plots.styling import add_style
from calculations.similarity import get_window_items, get_similar_items
from panel.viewable import Viewer


class TornadoPlot(Viewer):


    plot_single = param.ClassSelector(class_=figure)
    plot_overview = param.ClassSelector(class_=figure)

    def __init__(self, data, item, y_col, columns, feature_iter):
        all_selected_cols = feature_iter.all_selected_cols
        super().__init__()
        self.remaining_columns = [col for col in columns if col not in all_selected_cols]
        self.mean_prob = data[y_col].mean()

        # get all singular features
        single_dict = {}
        for col in columns:
            # get prediction of the col on its own
            single_dict[col] = get_window_items(data, item, col, y_col)[y_col].mean() - self.mean_prob

        self.dataset_single = get_dataset(data, item, y_col, self.remaining_columns, all_selected_cols, single_dict, self.mean_prob)
        self.dataset_overview = get_overview_dataset(data, item, y_col, columns, single_dict, self.mean_prob)
        self.plot_single = tornado_plot(self.dataset_single, feature_iter, "single")
        self.plot_overview = tornado_plot(self.dataset_overview, feature_iter, "overview")
        self.all_selected_cols = all_selected_cols
        self.feature_iter = feature_iter

    @param.depends('plot_overview')
    def __panel__(self):
        return self.plot_overview if len(self.all_selected_cols) == 0 else figure(width=0)



def get_dataset(data, item, y_col, remaining_columns, all_selected_cols, single_dict, mean_prob):

    # get all singular features
    results = []
    for col in remaining_columns:
        columns = all_selected_cols + [col]

        single_value = single_dict[col]

        if len(columns) > 1:
            first_col = all_selected_cols[0]

            # calculate added value
            similar_items = get_similar_items(data, item, all_selected_cols[1:])
            prev_value = get_window_items(similar_items, item, first_col, y_col)[y_col].mean() - mean_prob
            added_value = single_value + prev_value

            # calculate joined value
            # get prediction of the col with the other selected cols
            similar_items = get_similar_items(data, item, columns[1:])
            joined_value = get_window_items(similar_items, item, first_col, y_col)[y_col].mean() - mean_prob

            # value = single_mean - mean_prob
            value = np.abs(added_value - joined_value)

            results.append({'feature': col, 'prediction': joined_value, 'value': value})
        else:
            results.append({'feature': col, 'prediction': single_value, 'value': single_value})

    dataframe = create_dataframe(results)

    return dataframe


class InteractTree:
    value = 0
    prediction = 0
    added = 0
    prev_value = 0
    columns = []
    count = 0


class InteractTreeRoot (InteractTree):

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
                self.nodes.append(InteractTreeSub(single_dict, self, col, remaining, mean_prob, data, item, y_col, min_value))

    def get_nodes(self):
        nodes = []
        for node in self.nodes:
            nodes.extend(node.get_nodes())
        return nodes


class InteractTreeSub (InteractTree):
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
        self.value = (prev.value * prev.count + np.abs(joined_value - added_value)) / self.count
        #self.raw_value = np.abs(joined_value - added_value)
        #self.value = 0 if min_value/2 >= self.raw_value else (prev.value * prev.count + np.abs(joined_value - added_value)) / self.count
        self.nodes = []
        if len(self.columns) and self.value > min_value:
            for col in remaining_columns:
                remaining = [c for c in remaining_columns if c != col]
                self.nodes.append(InteractTreeSub(single_dict, self, col, remaining, mean_prob, data, item, y_col, min_value))

    def get_nodes(self):
        nodes = [self]
        for node in self.nodes:
            nodes.extend(node.get_nodes())
        return nodes


def get_overview_dataset(data, item, y_col, columns, single_dict, mean_prob):
    prob_range = data[y_col].max() - data[y_col].min()
    min_value = prob_range * 0.2

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
        results.append({'feature': ", ".join(node.columns), 'value': node.value, 'prediction': node.prediction, 'added': node.added, 'prev_value': node.prev_value})

    dataframe = create_dataframe(results)

    return dataframe


def create_dataframe(results):
    dataframe = pd.DataFrame(results)
    dataframe['abs_value'] = dataframe['value'].abs()
    dataframe['positive'] = dataframe['value'].apply(lambda x: 'pos' if x > 0 else 'neg')
    dataframe = dataframe.sort_values(by='abs_value', ascending=False)
    if len(dataframe) > 10:
        dataframe = dataframe.iloc[:10]
    dataframe = dataframe.sort_values(by='abs_value', ascending=True)
    return dataframe


def set_col(data, item_source, feature_iter, type):
    if len(item_source.selected.indices) > 0:
        if len(item_source.selected.indices) > 1:
            item_source.selected.indices = item_source.selected.indices[1:2]
        select = data.iloc[item_source.selected.indices]
        select = select['feature'].values[0]
        select = select.split(', ')
        if type == 'overview':
            feature_iter.set_all_selected_cols(select)
        else:
            feature_iter.add_col(select[0])
    else:
        feature_iter.set_all_selected_cols(data['feature'])


def tornado_plot(data, feature_iter, type):

    width = 400 if type == 'single' else 600

    item_source = ColumnDataSource(data=data)

    max_shap = data['abs_value'].max()
    min_shap = data['abs_value'].min()
    puffer = (max_shap - min_shap) * 0.1

    x_range = (min_shap - puffer, max_shap)

    plot = figure(title="Interaction strength", y_range=data['feature'], x_range=x_range, tools='tap', width=width,
                  toolbar_location=None)

    back_bars_left = plot.hbar(
        y='feature',
        right=plot.x_range.start,
        fill_color="white",
        line_width=0,
        source=item_source,
        nonselection_fill_alpha=0.0,
        selection_fill_alpha=1,
    )

    back_bars_right = plot.hbar(
        y='feature',
        right=plot.x_range.end,
        fill_color="white",
        line_width=0,
        source=item_source,
        nonselection_fill_alpha=0.0,
        selection_fill_alpha=1,
    )

    bars = plot.hbar(
        y='feature',
        right='abs_value',
        fill_color= "#3801AC",
        fill_alpha=1,
        nonselection_fill_alpha=1,
        line_width=0,
        source=item_source,
    )

    plot.xaxis.axis_label = "influence"

    plot = add_style(plot)

    plot.on_event('tap', lambda event: set_col(data, item_source, feature_iter, type))

    hover = HoverTool(renderers=[back_bars_left, back_bars_right], tooltips=[('@feature', '@value')])
    plot.add_tools(hover)

    return plot
