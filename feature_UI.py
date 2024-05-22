import bokeh.colors
import pandas as pd
import panel as pn
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.models import Band, ColumnDataSource
from bokeh.palettes import Blues9
import numpy as np
from scipy.stats import gaussian_kde
import functions as feature
import calculations.item_functions as item_functions
import calculations.column_functions as column_functions
import calculations.data_loader as data_loader
from plots.dependency_plot import dependency_scatterplot

pn.extension()

nn, means, testdata = data_loader.load_weather_data()

CLASSES = nn.classes_
COLUMNS = testdata.columns
data = testdata  # [0:200]
data_and_probabilities = feature.combine_data_and_results(data, nn, CLASSES)

# create widgets
x = pn.widgets.EditableIntSlider(name='x', start=0, end=199, value=26).servable()
col = pn.widgets.Select(name='column', options=[col for col in data.columns])
CHART_TYPE_OPTIONS = ['scatter', 'line', 'band', 'contour']
chart_type = pn.widgets.MultiChoice(name='chart_type', options=CHART_TYPE_OPTIONS, value=['scatter']).servable()

column_group = []
combined_columns = pn.widgets.LiteralInput(value=[])
num_groups = pn.widgets.LiteralInput(value=1)

row = pn.FlexBox().servable()
all_options = [name for name in COLUMNS]
remaining_options = pn.widgets.LiteralInput(value=[name for name in COLUMNS])


def get_group_options(index):
    # combine lists remaining_options with items in columngroup[index]
    return remaining_options.value.copy() + column_group[index].value.copy()


def callback(event):
    index = int(event.obj.name)
    combined_columns.value = column_functions.clean_column_groups(column_group)
    length = len(column_group[index].value)
    if length == 0 and index != (num_groups.value - 1):
        # remove old widget
        column_group.pop(index)
        row.pop(index)
        update_names()
        num_groups.value -= 1
    if length > 1 and (index == (num_groups.value - 1)):
        # add new widget
        column_group.append(
            pn.widgets.MultiChoice(name=str(num_groups.value), value=[], options=remaining_options.value.copy()))
        column_group[num_groups.value].param.watch(callback, parameter_names=['value'], onlychanged=False)
        row.append(column_group[num_groups.value])
        num_groups.value += 1

    # update remaining options
    columns = []
    for col in column_group:
        for name in col.value:
            columns.append(name)
    remaining_options.value = [name for name in all_options if name not in columns]
    for col in column_group:
        col.options = get_group_options(int(col.name))


def update_names():
    for i in range(num_groups.value - 1):
        column_group[i].name = str(i)


# add first columngroup widget
column_group.append(
    pn.widgets.MultiChoice(name=str(0), value=['Humidity9am', 'Humidity3pm'], options=remaining_options.value.copy()))
column_group[0].param.watch(callback, parameter_names=['value'], onlychanged=False)
row.append(column_group[0])
column_group[0].param.trigger('value') # trigger event to update remaining options

item_shap = pn.bind(item_functions.get_item_shap_values, testdata[0: 200], x, means, nn, COLUMNS, combined_columns)


def shap_tornado_plot(data):
    item_source = ColumnDataSource(data=data)
    chart2 = figure(title="example0", y_range=data['feature'], x_range=(-1, 1), tools='tap')
    chart2.hbar(
        y='feature',
        right='shap_value',
        fill_color=factor_cmap("positive", palette=["steelblue", "crimson"], factors=["pos", "neg"]),
        line_width=0,
        source=item_source,
        nonselection_fill_alpha=0.7,
        selection_hatch_pattern='horizontal_wave',
        selection_hatch_scale=7,
        selection_hatch_weight=1.5,
        selection_hatch_color='purple'
    )

    def set_col():
        if (len(item_source.selected.indices) > 0):
            if (len(item_source.selected.indices) > 1):
                item_source.selected.indices = item_source.selected.indices[1:2]
            select = data.iloc[item_source.selected.indices]
            select = select['feature'].values[0]
            col.value = select

    chart2.on_event('tap', set_col)
    return chart2


# displayed data
item_prediction = pn.bind(item_functions.get_item_prediction, data_and_probabilities, x)
prob_data = pn.bind(item_functions.get_item_probability_string, data_and_probabilities, x, item_prediction)
item_data = pn.bind(item_functions.get_item_data, data, x)

all_selected_cols = pn.bind(column_functions.return_col, col)
cur_feature = pn.widgets.Select(name='', options=all_selected_cols, align='center')

# displayed bokeh plots
shap_plot = pn.bind(shap_tornado_plot, item_shap)
dep_plot = pn.bind(dependency_scatterplot, data_and_probabilities, cur_feature, all_selected_cols,
                   item_prediction, x, chart_type)

# remaining layout
pn.panel(prob_data).servable()
pn.Row(item_data, shap_plot, pn.Column(dep_plot, cur_feature)).servable()
