import plotly.express as px
from palmerpenguins import load_penguins
from shiny import App, ui
from shinywidgets import output_widget, render_widget
import pandas as pd
import pickle
import functions as feature

penguins = load_penguins()


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
data = testdata[0:20]

data_and_probabilities = feature.combine_data_and_results(data, nn, classes)

app_ui = ui.page_fluid(
    ui.input_slider("n", "Number of bins", 1, 100, 20),
    output_widget("plot"),
)

def server(input, output, session):
    @render_widget
    def plot():
        scatterplot = px.bar(
            data_frame=data_and_probabilities,
            x="MaxTemp",
            y="prob_0",
        )

        return scatterplot

app = App(app_ui, server)