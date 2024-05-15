import altair as alt
import pandas
from vega_datasets import data
import panel as pn
import pickle

cars = data.cars()

penguins_url = "https://raw.githubusercontent.com/vega/vega/master/docs/data/penguins.json"

shap_weather = 0

def get_item(explanation, index):
    item = explanation.iloc[index]
    item = pandas.DataFrame({'feature': item.index, 'shap_value': item.values})
    #add column containing the absolute value of the shap value
    item['abs_shap_value'] = item['shap_value'].abs()
    #sort by the absolute value of the shap value
    item = item.sort_values(by='abs_shap_value', ascending=False)

    return item

with open('shap_weather.pkl', 'rb') as file:

    # Call load method to deserialze
    shap_weather = pickle.load(file)

    shap_weather_values = pandas.DataFrame(shap_weather.values, columns=shap_weather.feature_names)


    x = pn.widgets.IntSlider(name='x', start=0, end=199, value=26).servable()

    item = pn.bind(get_item, shap_weather_values, x)

    df_pane = pn.panel(item)
    df_pane.servable()

    def chart(data):
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
            height=300,
            width=100
        ).interactive()
        return chart1

    pn.extension('vega')

    pn.panel(pn.bind(chart,item)).servable()
