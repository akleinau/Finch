import panel as pn
import functions as feature
import calculations.item_functions as item_functions
import calculations.column_functions as column_functions
import calculations.data_loader as data_loader
from plots.dependency_plot import dependency_scatterplot
from plots.tornado_plot import shap_tornado_plot

pn.extension()

nn, raw_data = data_loader.load_weather_data()
means = feature.get_means(raw_data)

CLASSES = nn.classes_
COLUMNS = raw_data.columns
data = raw_data  # [0:200]
data_and_probabilities = feature.combine_data_and_results(data, nn, CLASSES)

# create widgets
x = pn.widgets.EditableIntSlider(name='x', start=0, end=199, value=26).servable()
col = pn.widgets.Select(name='column', options=[col for col in data.columns])
CHART_TYPE_OPTIONS = ['scatter', 'line', 'band', 'contour']
chart_type = pn.widgets.MultiChoice(name='chart_type', options=CHART_TYPE_OPTIONS, value=['scatter']).servable()

# create all the widgets and variables needed for the column group selection
column_group = []
combined_columns = pn.widgets.LiteralInput(value=[])
num_groups = pn.widgets.LiteralInput(value=1)

row = pn.FlexBox().servable()
all_options = [name for name in COLUMNS]
remaining_options = pn.widgets.LiteralInput(value=[name for name in COLUMNS])

# add first columngroup widget
column_group.append(
    pn.widgets.MultiChoice(name=str(0), value=['Humidity9am', 'Humidity3pm'], options=remaining_options.value.copy()))
widget = [column_group, row, num_groups, remaining_options, combined_columns, all_options]
column_group[0].param.watch(lambda event: column_functions.column_group_changed(event, widget),
                            parameter_names=['value'], onlychanged=False)
row.append(column_group[0])
column_group[0].param.trigger('value') # trigger event to update remaining options

# display prediction
item_prediction = pn.bind(item_functions.get_item_prediction, data_and_probabilities, x)
prob_data = pn.bind(item_functions.get_item_probability_string, data_and_probabilities, x, item_prediction)
item_data = pn.bind(item_functions.get_item_data, data, x)

# display shap plot
item_shap = pn.bind(item_functions.get_item_shap_values, data, x, means, nn, COLUMNS, combined_columns)
shap_plot = pn.bind(shap_tornado_plot, item_shap, [col])  # col is wrapped to be passed as reference

# display dependency plot
all_selected_cols = pn.bind(column_functions.return_col, col)
cur_feature = pn.widgets.Select(name='', options=all_selected_cols, align='center')
dep_plot = pn.bind(dependency_scatterplot, data_and_probabilities, cur_feature, all_selected_cols,
                   item_prediction, x, chart_type)

# remaining layout
pn.panel(prob_data).servable()
pn.Row(item_data, shap_plot, pn.Column(dep_plot, cur_feature)).servable()
