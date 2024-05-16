import pandas as pd
import panel as pn
import pickle
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.models import ColumnDataSource

def get_item_shap_values(explanation, index):
    item = explanation.iloc[index]
    item = pd.DataFrame({'feature': item.index, 'shap_value': item.values})
    #add column containing the absolute value of the shap value
    item['abs_shap_value'] = item['shap_value'].abs()
    item['positive'] = item['shap_value'].map(lambda x: 'pos' if x > 0 else 'neg')
    #sort by the absolute value of the shap value
    item = item.sort_values(by='abs_shap_value', ascending=True)

    return item

def get_item_data(explanation, index):
    item = explanation.iloc[index]
    item = pd.DataFrame({'feature': item.index, 'value': item.values})
    return item

with open('weather_result.pkl', 'rb') as file:

    # prepare the data
    shap_weather = pickle.load(file)

    columns_data = [name for name in shap_weather.columns if not (name.startswith('shap_') | name.startswith('prob_'))]
    columns_shap = [name for name in shap_weather.columns if name.startswith('shap_')]

    shap_weather_values = shap_weather[columns_shap]
    weather_data = shap_weather[columns_data]


    #create widgets
    x = pn.widgets.IntSlider(name='x', start=0, end=199, value=26).servable()
    col = pn.widgets.Select(name='column', options=[col for col in shap_weather.columns])

    item_shap = pn.bind(get_item_shap_values, shap_weather_values, x)

    def chart2(data):
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
                col.value = select[5:]

        chart2.on_event('tap', setCol)
        return chart2

    def chart3(data, col, prob, index):
        item = data.iloc[index]
        chart3 = figure(title="example", x_axis_label=col, y_axis_label=prob, tools='tap')
        chart3.scatter(data[col], data[prob], color='forestgreen')
        chart3.scatter(item[col], item[prob], color='purple', size=7)
        return chart3

    def probability(data, index, prob):
        return data.iloc[index][prob]

    def prediction(data, index):
        return data.iloc[index]['prediction']


    #displayed data
    item_prediction = pn.bind(prediction, shap_weather, x)
    prob_data = pn.bind(probability, shap_weather, x, item_prediction)
    item_data = pn.bind(get_item_data, weather_data, x)

    #displayed bokeh plots
    shap_plot = pn.bind(chart2, item_shap)
    dep_plot = pn.bind(chart3, shap_weather, col, item_prediction, x)

    #remaining layout
    pn.panel(item_prediction).servable()
    pn.panel(prob_data).servable()
    pn.Row(item_data, shap_plot, dep_plot).servable()


