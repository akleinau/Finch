import pandas as pd
import panel as pn
import pickle
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.models import ColumnDataSource
import functions as feature

pn.extension()

def get_item_shap_values(explanation, index, means, nn, COLUMNS, combined_columns=None):
    item = explanation.iloc[[index]]
    shap_explanations = feature.calc_shap_values(item, means, nn, COLUMNS, combined_columns)
    shap_values = pd.DataFrame(shap_explanations.values,
                                   columns=shap_explanations.feature_names)
    #pivot the data, so that each row contains the feature and the shap value
    shap_values = shap_values.melt(var_name='feature', value_name='shap_value')

    #add column containing the absolute value of the shap value
    shap_values['abs_shap_value'] = shap_values['shap_value'].abs()
    shap_values['positive'] = shap_values['shap_value'].map(lambda x: 'pos' if x > 0 else 'neg')
    #sort by the absolute value of the shap value
    combined_item = shap_values.sort_values(by='abs_shap_value', ascending=True)

    return combined_item

def get_item_data(explanation, index):
    item = explanation.iloc[index]
    item = pd.DataFrame({'feature': item.index, 'value': item.values})
    return item

# load weather files
file_nn = open('weather_nn.pkl', 'rb')
nn = pickle.load(file_nn)
file_means = open('weather_means.pkl', 'rb')
means = pickle.load(file_means)
#file.close()
file_testdata = open('weather_testdata.pkl', 'rb')
testdata = pickle.load(file_testdata)
file_nn.close()
file_means.close()

classes = nn.classes_
columns = testdata.columns
data = testdata[0:200]

data_and_probabilities = feature.combine_data_and_results(data, nn, classes)

#create widgets
x = pn.widgets.IntSlider(name='x', start=0, end=199, value=26).servable()
col = pn.widgets.Select(name='column', options=[col for col in data.columns])

columngroup = []
combined_columns = pn.widgets.LiteralInput(value=[])
num_groups = pn.widgets.LiteralInput(value=1)

row = pn.Row().servable()
all_options = [name for name in columns]
remaining_options = pn.widgets.LiteralInput(value=[name for name in columns])

#watcher = num_groups.param.watch(callbackNum, parameter_names=['value'], onlychanged=False)

def clean_column_groups(group):
    columns = []
    for col in group:
        if (len(col.value) > 1):
            columns.append(col.value)

    return columns

def get_group_options(index):
    #combine lists remaining_options with items in columngroup[index]
    return remaining_options.value.copy() + columngroup[index].value.copy()

def callback(event):
    index = int(event.obj.name)
    combined_columns.value = clean_column_groups(columngroup)
    length = len(columngroup[index].value)
    if length == 0 and index != (num_groups.value - 1):
        #remove old widget
        columngroup.pop(index)
        row.pop(index)
        updateNames()
        num_groups.value -= 1
    if length > 1 and (index == (num_groups.value - 1)):
        #add new widget
        columngroup.append(pn.widgets.MultiChoice(name=str(num_groups.value), value=[], options=remaining_options.value.copy()))
        watcher = columngroup[num_groups.value].param.watch(callback, parameter_names=['value'], onlychanged=False)
        row.append(columngroup[num_groups.value])
        num_groups.value += 1

    # update remaining options
    columns = []
    for col in columngroup:
        for name in col.value:
            columns.append(name)
    remaining_options.value = [name for name in all_options if name not in columns]
    for col in columngroup:
        col.options = get_group_options(int(col.name))

def updateNames():
    for i in range(num_groups.value - 1):
        columngroup[i].name = str(i)

for i in range(num_groups.value):
    columngroup.append(pn.widgets.MultiChoice(name=str(i), value=[], options=remaining_options.value.copy()))
    watcher = columngroup[i].param.watch(callback, parameter_names=['value'], onlychanged=False)
    row.append(columngroup[i])

item_shap = pn.bind(get_item_shap_values, testdata[0: 200], x, means, nn, columns, combined_columns)

def shap_tornado_plot(data):
    item_source = ColumnDataSource(data=data)
    chart2 = figure(title="example0", y_range=data['feature'], x_range=(-1, 1), tools='tap')
    chart2.hbar(y='feature', right='shap_value', fill_color=factor_cmap("positive", palette=["steelblue", "crimson"], factors=["pos", "neg"]), line_width=0, source=item_source)

    def setCol():
        select = data.iloc[item_source.selected.indices]
        if (len(select) == 0):
            select = ""
        else:
            select = select['feature'].values[0]
            #change the value of the col widget
            col.value = select

    chart2.on_event('tap', setCol)
    return chart2

def dependency_scatterplot(data, combined_col, prob, index):
    item = data.iloc[index]
    col = combined_col.split(", ")[0]
    chart3 = figure(title="example", x_axis_label=col, y_axis_label=prob, tools='tap')
    chart3.scatter(data[col], data[prob], color='forestgreen', alpha=0.5)
    chart3.scatter(item[col], item[prob], color='purple', size=7)
    return chart3

def probability(data, index, prob):
    return prob + " with probability: " + "{:10.2f}".format(data.iloc[index][prob])

def prediction(data, index):
    return data.iloc[index]['prediction']


#displayed data
item_prediction = pn.bind(prediction, data_and_probabilities, x)
prob_data = pn.bind(probability, data_and_probabilities, x, item_prediction)
item_data = pn.bind(get_item_data, data, x)

#displayed bokeh plots
shap_plot = pn.bind(shap_tornado_plot, item_shap)
dep_plot = pn.bind(dependency_scatterplot, data_and_probabilities, col, item_prediction, x)

#remaining layout
pn.panel(prob_data).servable()
pn.Row(item_data, shap_plot, dep_plot).servable()


