import pandas as pd
import functions as feature


def get_item_shap_values(explanation, index, means, nn, COLUMNS, combined_columns=None):
    item = explanation.iloc[[index]]
    shap_explanations = feature.calc_shap_values(item, means, nn, COLUMNS, combined_columns)
    shap_values = pd.DataFrame(shap_explanations.values,
                               columns=shap_explanations.feature_names)
    # pivot the data, so that each row contains the feature and the shap value
    shap_values = shap_values.melt(var_name='feature', value_name='shap_value')

    # add column containing the absolute value of the shap value
    shap_values['abs_shap_value'] = shap_values['shap_value'].abs()
    shap_values['positive'] = shap_values['shap_value'].map(lambda x: 'pos' if x > 0 else 'neg')
    # sort by the absolute value of the shap value
    combined_item = shap_values.sort_values(by='abs_shap_value', ascending=True)

    return combined_item


def get_item_data(explanation, index):
    item = explanation.iloc[index]
    item = pd.DataFrame({'feature': item.index, 'value': item.values})
    return item


def get_item_probability_string(data, index, prob):
    return prob + " with probability: " + "{:10.2f}".format(data.iloc[index][prob])


def get_item_prediction(data, index):
    return data.iloc[index]['prediction']
