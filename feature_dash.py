from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import pickle
import functions as feature


df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')


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

app = Dash()

app.layout = [
    html.H1(children='Title of Dash App', style={'textAlign':'center'}),
    dcc.Dropdown(df.country.unique(), 'Canada', id='dropdown-selection'),
    dcc.Graph(id='graph-content')
]

@callback(
    Output('graph-content', 'figure'),
    Input('dropdown-selection', 'value')
)



def update_graph(value):
    scatter = px.scatter(data_and_probabilities, x='MaxTemp', y='prob_0')

    return scatter

if __name__ == '__main__':
    app.run(debug=True)
