import altair as alt
import pandas
import panel as pn
import pickle

shap_weather = 0

def get_item_shap_values(explanation, index):
    item = explanation.iloc[index]
    item = pandas.DataFrame({'feature': item.index, 'shap_value': item.values})
    #add column containing the absolute value of the shap value
    item['abs_shap_value'] = item['shap_value'].abs()
    #sort by the absolute value of the shap value
    item = item.sort_values(by='abs_shap_value', ascending=False)

    return item

def get_item_data(explanation, index):
    item = explanation.iloc[index]
    item = pandas.DataFrame({'feature': item.index, 'value': item.values})
    return item

with open('weather_result.pkl', 'rb') as file:

    # Call load method to deserialze
    shap_weather = pickle.load(file)

    columns_data = [name for name in shap_weather.columns if not (name.startswith('shap_') | name.startswith('prob_'))]
    columns_shap = [name for name in shap_weather.columns if name.startswith('shap_')]

    shap_weather_values = shap_weather[columns_shap]
    weather_data = shap_weather[columns_data]


    x = pn.widgets.IntSlider(name='x', start=0, end=199, value=26).servable()
    col = pn.widgets.Select(name='column', options=[col for col in shap_weather.columns]).servable()
    prob = pn.widgets.Select(name='column', options=[col for col in shap_weather.columns if col.startswith('prob_')]).servable()

    item = pn.bind(get_item_shap_values, shap_weather_values, x)

    def chart1(data):
        chart1 = alt.Chart(data).mark_bar().encode(
            y=alt.Y('feature').sort(),
            x=alt.X('shap_value', scale=alt.Scale(domain=(-1, 1))),
            tooltip=['feature', 'shap_value'],
            color=alt.condition(
                alt.datum.shap_value > 0,
                alt.value("steelblue"),  # The positive color
                alt.value("crimson")  # The negative color
            )
        ).properties(
            height=400,
            width=200
        ).interactive()
        return chart1

    def chart2(data, col, prob):
        chart2 = alt.Chart(data).mark_point().encode(
            y=alt.Y(prob),
            x=alt.X(col),
        ).properties(
            height=400,
            width=200
        ).interactive()
        return chart2

    pn.extension('vega')

    shap_plot = pn.bind(chart1,item)

    item_data = pn.bind(get_item_data, weather_data, x)

    dep_plot = pn.bind(chart2,shap_weather, col, prob)

    pn.Row(item_data, shap_plot, dep_plot).servable()


