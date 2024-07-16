import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.models import ColumnDataSource, HoverTool
import param

from plots.styling import add_style
from calculations.similarity import get_window_items, get_similar_items
from panel.viewable import Viewer
import panel as pn


class TornadoPlot(Viewer):


    plot_single = param.ClassSelector(class_=figure)
    panel_single = param.ClassSelector(class_=pn.Column)
    plot_overview = param.ClassSelector(class_=figure)
    ranked_buttons = param.ClassSelector(class_=pn.FlexBox)
    all_selected_cols = param.List()

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
        self.plot_single = self.tornado_plot(self.dataset_single, feature_iter, "single")
        self.panel_single = pn.Column("## Strength of interactions with other features: ", self.plot_single)
        self.plot_overview = self.tornado_plot(self.dataset_overview, feature_iter, "overview")
        self.ranked_buttons = self.create_ranked_buttons(self.dataset_overview, feature_iter)
        self.all_selected_cols = all_selected_cols
        self.feature_iter = feature_iter
        self.ranked_buttons_text = "## (advanced) fast selection:"

    @param.depends('ranked_buttons')
    def __panel__(self):
        return pn.Column(self.ranked_buttons_text, self.ranked_buttons) if len(self.all_selected_cols) == 0 \
            else None

    @param.depends('ranked_buttons')
    def get_panel_single(self):
        return self.panel_single if len(self.all_selected_cols) == 0 else None

    def hide_all(self):
        self.all_selected_cols = []
        self.ranked_buttons_text = ""
        self.panel_single = None
        self.ranked_buttons = pn.FlexBox()

    def set_col(self, data, item_source, feature_iter, type):
        if len(item_source.selected.indices) > 0:

            self.hide_all()

            if len(item_source.selected.indices) > 1:
                item_source.selected.indices = item_source.selected.indices[1:2]
            select = data.iloc[item_source.selected.indices]
            select = select['feature'].values[0]
            if type == 'overview':
                select = select.split(', ')
                feature_iter.set_all_selected_cols(select)
            else:
                select = select.split(' = ')
                feature_iter.add_col(select[0])
        else:
            feature_iter.set_all_selected_cols(data['feature'])


    def tornado_plot(self, data, feature_iter, type):

        if len(data) == 0:
            return figure()

        width = 350 if type == 'single' else 600

        item_source = ColumnDataSource(data=data)

        max_shap = data['abs_value'].max()
        min_shap = data['abs_value'].min()
        puffer = (max_shap - min_shap) * 0.1

        x_range = (min_shap - puffer, max_shap)

        plot = figure(title="", y_range=data['feature'], x_range=x_range, tools='tap', width=width,
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
            fill_color= "#381eaa",
            fill_alpha=1,
            nonselection_fill_alpha=1,
            line_width=0,
            source=item_source,
        )

        plot.xaxis.axis_label = "influence"

        plot = add_style(plot)

        plot.on_event('tap', lambda event: self.set_col(data, item_source, feature_iter, type))

        hover = HoverTool(renderers=[back_bars_left, back_bars_right], tooltips=[('', '@feature')])

        # hide x axis
        #plot.xaxis.visible = False

        plot.add_tools(hover)

        return plot

    def create_ranked_buttons(self, data, feature_iter):
        buttons = []
        for i, row in data.iterrows():
            button = pn.widgets.Button(name=row['feature'], button_type='default')
            button.on_click(lambda event: self.ranked_buttons_clicked(event, feature_iter))
            buttons.append(button)
        return pn.FlexBox(*buttons)

    def ranked_buttons_clicked(self, event, feature_iter):
        self.hide_all()
        feature_iter.set_all_selected_cols(event.obj.name.split(', '))


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

            results.append({'feature': col, 'prediction': joined_value, 'value': value,
                            'item_value': item.data_raw[col].values[0]})
        else:
            results.append({'feature': col, 'prediction': single_value, 'value': single_value,
                            'item_value': item.data_raw[col].values[0]})

    dataframe = create_dataframe(results)

    return dataframe


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
        results.append({'feature': ", ".join(node.columns), 'value': node.value, 'prediction': node.prediction,
                        'added': node.added, 'prev_value': node.prev_value})

    dataframe = create_dataframe(results)

    return dataframe

def create_dataframe(results):
    dataframe = pd.DataFrame(results)
    if len(dataframe) == 0:
        return dataframe

    dataframe['abs_value'] = dataframe['value'].abs()
    dataframe['positive'] = dataframe['value'].apply(lambda x: 'pos' if x > 0 else 'neg')
    dataframe = dataframe.sort_values(by='abs_value', ascending=False)
    if len(dataframe) > 10:
        dataframe = dataframe.iloc[:10]
    dataframe = dataframe.sort_values(by='abs_value', ascending=True)

    if "item_value" in dataframe.columns:
        dataframe['feature'] = dataframe['feature'] + " = " + dataframe['item_value'].apply(lambda x: str(round(x, 2)))

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