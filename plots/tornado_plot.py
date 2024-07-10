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
        self.dataset = get_dataset(data, item, y_col, self.remaining_columns, all_selected_cols)
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

    print(dataframe)

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

    plot = figure(title="Interaction strength", y_range=data['feature'], x_range=x_range, tools='tap', width=400,
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

    hover = HoverTool(renderers=[back_bars_left, back_bars_right], tooltips=[('', '@feature')])
    plot.add_tools(hover)

    return plot
