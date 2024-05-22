import bokeh.colors
import pandas as pd
import panel as pn
import pickle
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.models import Band, ColumnDataSource
from bokeh.palettes import Blues9
import numpy as np
from scipy.stats import gaussian_kde
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
data = testdata

data_and_probabilities = feature.combine_data_and_results(data, nn, classes)

#create widgets
x = pn.widgets.EditableIntSlider(name='x', start=0, end=199, value=26).servable()
col = pn.widgets.Select(name='column', options=[col for col in data.columns])
chart_type_options = ['scatter', 'line', 'band', 'contour']
chart_type = pn.widgets.MultiChoice(name='chart_type', options=chart_type_options, value=['scatter']).servable()

columngroup = []
combined_columns = pn.widgets.LiteralInput(value=[])
num_groups = pn.widgets.LiteralInput(value=1)

row = pn.FlexBox().servable()
all_options = [name for name in columns]
remaining_options = pn.widgets.LiteralInput(value=[name for name in columns])

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

#add first columngroup widget
columngroup.append(pn.widgets.MultiChoice(name=str(0), value=['Humidity9am', 'Humidity3pm'], options=remaining_options.value.copy()))
watcher = columngroup[0].param.watch(callback, parameter_names=['value'], onlychanged=False)
row.append(columngroup[0])
#trigger event to update remaining options
columngroup[0].param.trigger('value')


item_shap = pn.bind(get_item_shap_values, testdata[0: 200], x, means, nn, columns, combined_columns)

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

    def setCol():
        if (len(item_source.selected.indices) > 0):
            if (len(item_source.selected.indices) > 1):
                item_source.selected.indices = item_source.selected.indices[1:2]
            select = data.iloc[item_source.selected.indices]
            select = select['feature'].values[0]
            col.value = select

    chart2.on_event('tap', setCol)
    return chart2

def dependency_scatterplot(data, col, all_selected_cols, prob, index, chart_type):
    item = data.iloc[index]
    sorted_data = data.sort_values(by=col)

    if (len(all_selected_cols) > 1):
        item_val = item[all_selected_cols[1]]
        sorted_data["scatter_group"] = sorted_data[all_selected_cols[1]].apply(lambda x: 'saddlebrown' if x >= item_val else 'midnightblue')
    else:
        sorted_data["scatter_group"] = 'forestgreen'

    x_range = (sorted_data[col].min(), sorted_data[col].max())

    chart3 = figure(title="example", y_axis_label=prob, tools='tap', y_range=(0,1), x_range=x_range)
    chart3.grid.level = "overlay"
    chart3.grid.grid_line_color = "black"
    chart3.grid.grid_line_alpha = 0.05

    #for the contours
    def kde(x, y, N):
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()

        X, Y = np.mgrid[xmin:xmax:N * 1j, ymin:ymax:N * 1j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([x, y])
        kernel = gaussian_kde(values)
        Z = np.reshape(kernel(positions).T, X.shape)

        return X, Y, Z

    #create bands and contours
    colors = ['midnightblue', 'saddlebrown', 'forestgreen']
    for i, color in enumerate(colors):
        filtered_data = sorted_data[sorted_data["scatter_group"] == color].sort_values(by=col)
        if (len(filtered_data) > 0):
            window = len(filtered_data) // 20
            rolling = filtered_data[prob].rolling(window=window, center=True).agg(['mean', 'std'])
            rolling['upper'] = rolling['mean'] + rolling['std']
            rolling['lower'] = rolling['mean'] - rolling['std']
            combined = pd.concat([filtered_data, rolling], axis=1)
            combined = ColumnDataSource(combined.reset_index())

            if "line" in chart_type:
                chart3.line(col, 'mean', source=combined, color=color, line_width=2)

            if "band" in chart_type:
                band = Band(base=col, lower='lower', upper='upper', source=combined,
                        fill_color=color)

                chart3.add_layout(band)

            if "contour" in chart_type:
                #contours
                #only use subset of data for performance reasons
                subsetted_data = filtered_data.sample(n=1000)
                x,y,z = kde(subsetted_data[col], subsetted_data[prob], 100)

                #use the color to create a palette
                cur_color =  bokeh.colors.named.NamedColor.find(color)
                palette = [cur_color]
                for i in range(0, 3):
                    palette.append(palette[i].lighten(0.2))
                #convert to hex
                palette = [c.to_hex() for c in palette]
                #invert the palette
                palette = palette[::-1]


                levels = np.linspace(np.min(z), np.max(z), 7)
                chart3.contour(x, y, z, levels[1:], fill_color=palette, line_color=palette, fill_alpha=0.8)


    if "scatter" in chart_type:
        alpha = 0.3
        chart3.scatter(sorted_data[col], sorted_data[prob], color=sorted_data["scatter_group"],
                       alpha=alpha, marker='dot', size=7)
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

def return_col(combined_col):
    col = combined_col.split(", ")
    return [c for c in col]

all_selected_cols = pn.bind(return_col, col)
cur_feature = pn.widgets.Select(name='', options=all_selected_cols, align='center')

#displayed bokeh plots
shap_plot = pn.bind(shap_tornado_plot, item_shap)
dep_plot = pn.bind(dependency_scatterplot, data_and_probabilities, cur_feature, all_selected_cols, item_prediction, x, chart_type)



#remaining layout
pn.panel(prob_data).servable()
pn.Row(item_data, shap_plot, pn.Column(dep_plot, cur_feature)).servable()


