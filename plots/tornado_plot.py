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


    plot = param.ClassSelector(class_=figure)

    def __init__(self, data, item, y_col, columns, all_selected_cols):
        super().__init__()
        self.remaining_columns = [col for col in columns if col not in all_selected_cols]
        #self.dataset = get_dataset(data, item, y_col, self.remaining_columns, all_selected_cols)
        self.dataset = get_overview_dataset(data, item, y_col, columns)
        self.plot = tornado_plot(self.dataset, all_selected_cols)

    @param.depends('plot')
    def __panel__(self):
        return self.plot



def get_dataset(data, item, y_col, remaining_columns, all_selected_cols):


    mean_prob = data[y_col].mean()

    # get all singular features
    results = []
    for col in remaining_columns:
        columns = all_selected_cols + [col]

        # get prediction of the col on its own
        single_mean = get_window_items(data, item, col, y_col)[y_col].mean()

        if len(columns) > 1:
            # get prediction of the col with the other selected cols
            similar_items = get_similar_items(data, item, columns)
            group_mean = get_window_items(similar_items, item, col, y_col)[y_col].mean()

            # get prediction of the other selected cols on their own
            similar_items = get_similar_items(data, item, all_selected_cols)
            prev_mean = get_window_items(similar_items, item, col, y_col)[y_col].mean()

            # value = single_mean - mean_prob
            value = np.abs(group_mean - (prev_mean + single_mean - mean_prob))

            results.append({'feature': col, 'prediction': group_mean, 'value': value})
        else:
            results.append({'feature': col, 'prediction': single_mean, 'value': single_mean- mean_prob})



    dataframe = pd.DataFrame(results).sort_values(by='value', ascending=False)
    dataframe['abs_value'] = dataframe['value'].abs()
    dataframe['positive'] = dataframe['value'].apply(lambda x: 'pos' if x > 0 else 'neg')
    dataframe = dataframe.sort_values(by='abs_value', ascending=True)

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
        nodes = [self]
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


def get_overview_dataset(data, item, y_col, columns):
    print("start calculation")

    mean_prob = data[y_col].mean()
    prob_range = data[y_col].max() - data[y_col].min()
    min_value = prob_range * 0.2

    # first, only keep a list of features independent of each other
    delete_dependent = False
    if delete_dependent:
        corr = data[columns].corr()
        independent_columns = []
        for col in columns:
            # check if the column is independent of the other columns
            if all([abs(corr[col][c]) < 0.7 for c in independent_columns]):
                independent_columns.append(col)

        deleted_columns = [col for col in columns if col not in independent_columns]
        print("deleted columns: ", deleted_columns)
    else:
        independent_columns = columns

    # get all singular features
    single_dict = {}
    for col in independent_columns:
        # get prediction of the col on its own
        single_dict[col] = get_window_items(data, item, col, y_col)[y_col].mean() - mean_prob

    print(single_dict)

    # recursively build a tree for each column
    tree_list = []
    for col in independent_columns:
        remaining = [c for c in independent_columns if c != col]
        tree_list.append(InteractTreeRoot(single_dict, col, remaining, mean_prob, data, item, y_col, min_value))

    # create a list of all nodes
    nodes = []
    for tree in tree_list:
        nodes.extend(tree.get_nodes())

    # create a dataframe from the nodes
    results = []
    for node in nodes:
        results.append({'feature': ", ".join(node.columns), 'value': node.value, 'prediction': node.prediction, 'added': node.added, 'prev_value': node.prev_value})

    dataframe = pd.DataFrame(results)
    dataframe['abs_value'] = dataframe['value'].abs()
    dataframe['positive'] = dataframe['value'].apply(lambda x: 'pos' if x > 0 else 'neg')
    dataframe = dataframe.sort_values(by='abs_value', ascending=False)

    print(dataframe)

    if len(dataframe) > 10:
        dataframe = dataframe.iloc[:10]

    dataframe = dataframe.sort_values(by='abs_value', ascending=True)

    return dataframe



def set_col(data, item_source, col):
    if len(item_source.selected.indices) > 0:
        if len(item_source.selected.indices) > 1:
            item_source.selected.indices = item_source.selected.indices[1:2]
        select = data.iloc[item_source.selected.indices]
        select = select['feature'].values[0]
        col[0].value = select  # col[0], bc the widget had to be wrapped in a list to be changed
    else:
        col[0].value = ", ".join(data['feature'])


def tornado_plot(data, col):
    item_source = ColumnDataSource(data=data)
    #get last item
    #col[0].value = shap['feature'].values[-1]
    max_shap = data['value'].max()
    min_shap = data['value'].min()

    x_range = (-1, 1) if type != 'global' else (0, 1.1*max_shap)
    if (max_shap > 1 or min_shap < -1):
        x_range = (min_shap, max_shap)

    plot = figure(title="Interaction strength", y_range=data['feature'], x_range=x_range, tools='tap', width=800,
                  toolbar_location=None)

    back_bars_left = plot.hbar(
        y='feature',
        right=plot.x_range.start,
        fill_color="lavender",
        line_width=0,
        source=item_source,
        nonselection_fill_alpha=0.0,
        selection_fill_alpha=1,
    )

    back_bars_right = plot.hbar(
        y='feature',
        right=plot.x_range.end,
        fill_color="lavender",
        line_width=0,
        source=item_source,
        nonselection_fill_alpha=0.0,
        selection_fill_alpha=1,
    )

    bars = plot.hbar(
        y='feature',
        right='value',
        fill_color=factor_cmap("positive", palette=["#AE0139", "#3801AC"], factors=["pos", "neg"]),
        fill_alpha=1,
        nonselection_fill_alpha=1,
        line_width=0,
        source=item_source,
    )

    plot.xaxis.axis_label = "influence"

    plot = add_style(plot)

    #plot.on_event('tap', lambda event: set_col(data, item_source, col))
    #set_col(data, item_source, col)

    hover = HoverTool(renderers=[back_bars_left, back_bars_right], tooltips=[('@feature', '@value')])
    plot.add_tools(hover)

    return plot
