import panel as pn
import functions as feature
import calculations.item_functions as item_functions
import calculations.column_functions as column_functions
import calculations.data_loader as data_loader
import calculations.similarity as similarity
from plots.dependency_plot import dependency_scatterplot
from plots.tornado_plot import shap_tornado_plot

pn.extension()

file_input_data = pn.widgets.FileInput(accept='.csv', name='Upload data')
file_input_nn = pn.widgets.FileInput(accept='.pkl', name='Upload neural network')

raw_data = pn.bind(data_loader.load_data, file_input_data, file_input_nn)
nn = pn.bind(data_loader.load_nn, file_input_nn, file_input_data)
means = pn.bind(feature.get_means, raw_data)

CLASSES = pn.bind(lambda nn: nn.classes_, nn)
COLUMNS = pn.bind(lambda data: [col for col in data.columns], raw_data)
data = pn.bind(lambda data: data[0:1000], raw_data)
data_and_probabilities = pn.bind(feature.combine_data_and_results, data, nn, CLASSES)

# create widgets
x = pn.widgets.EditableIntSlider(name='x', start=0, end=100, value=26)
pn.Row(file_input_data, file_input_nn, x).servable()
col = pn.widgets.Select(name='column', options=COLUMNS)
CHART_TYPE_OPTIONS = ['scatter', 'line', 'band', 'contour']
chart_type = pn.widgets.MultiChoice(name='chart_type', options=CHART_TYPE_OPTIONS, value=['line']).servable()

# create all the widgets and variables needed for the column group selection
column_group = []
combined_columns = pn.widgets.LiteralInput(value=[])
num_groups = pn.widgets.LiteralInput(value=1)
pn.panel("<br>").servable()
pn.panel("### Grouped columns:").servable()
row = pn.FlexBox().servable()
remaining_options = pn.widgets.LiteralInput(value=COLUMNS)
widget = [column_group, row, num_groups, remaining_options, combined_columns, COLUMNS]
column_functions.init_groups(widget)

# display prediction
item_prediction = pn.bind(item_functions.get_item_prediction, data_and_probabilities, x)
prob_data = pn.bind(item_functions.get_item_probability_string, data_and_probabilities, x, item_prediction)
item_data = pn.bind(item_functions.get_item_data, data, x)

# similarity experiments
test = pn.bind(similarity.test_setup, data_and_probabilities, COLUMNS, item_prediction)

# display shap plot
item_shap = pn.bind(item_functions.get_item_shap_values, data, x, means, nn, COLUMNS, combined_columns)
shap_plot = pn.bind(shap_tornado_plot, item_shap, [col])  # col is wrapped to be passed as reference

# display dependency plot
all_selected_cols = pn.bind(column_functions.return_col, col)
cur_feature = pn.widgets.Select(name='', options=all_selected_cols, align='center')
clustered_data = pn.bind(similarity.get_tree_groups, data_and_probabilities, all_selected_cols, cur_feature, item_prediction)
dep_plot = pn.bind(dependency_scatterplot, clustered_data, cur_feature, all_selected_cols,
                   item_prediction, x, chart_type)

#update everything when the data changes
file_input_data.param.watch(lambda event: data_loader.data_changed(event, [col, cur_feature, all_selected_cols, widget]), parameter_names=['value'], onlychanged=False)
file_input_nn.param.watch(lambda event: data_loader.data_changed(event, [col, cur_feature, all_selected_cols, widget]), parameter_names=['value'], onlychanged=False)

# remaining layout
pn.pane.Str(prob_data, sizing_mode="stretch_width", align="center", styles={"font-size":"20px", "text-align": "center"}).servable()
pn.Row(item_data, shap_plot, pn.Column(dep_plot, cur_feature)).servable()