import pandas as pd

from calculations import shap_set_functions
import panel as pn


class Item:
    def __init__(self, data_loader, data_and_probabilities, index, combined_columns=None):
        self.prediction = get_item_prediction(data_and_probabilities, index)
        self.shap = get_item_shap_values(data_loader, index, combined_columns)
        self.data = get_item_data(data_loader.data, index)
        self.prob_data =get_item_probability_string(data_and_probabilities, index, self.prediction)
        self.prob_wo_selected_cols = get_prob_wo_selected_cols(data_loader.nn, data_loader.columns, data_loader.means, self.data, self.prediction)


    def prediction_string(self):
        return pn.pane.Str(self.prob_data, sizing_mode="stretch_width", align="center",
                    styles={"font-size": "20px", "text-align": "center"})

    def table(self):
        return self.data


def get_item_shap_values(data_loader, index, combined_columns=None):
    item = data_loader.data.iloc[[index]]
    shap_explanations = shap_set_functions.calc_shap_values(item, data_loader.means, data_loader.nn, data_loader.columns, combined_columns)
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
    return "Prediction: " + prob[5:] + "  " + "{:10.0f}".format(data.iloc[index][prob] * 100) + "% certainty"


def get_item_prediction(data, index):
    return data.iloc[index]['prediction']


def get_prob_wo_selected_cols(nn, all_selected_cols, means, item, pred_label):
    item_df = pd.DataFrame(item['value'].values, index=item['feature'].values).T
    new_item = means.copy()

    # replace the values of the selected columns with the mean
    for col in all_selected_cols:
        new_item[col] = item_df[col].iloc[0]

    # calculate the prediction without the selected columns
    prediction = nn.predict_proba(new_item)
    prediction = pd.DataFrame(prediction, columns=[str(a) for a in nn.classes_])
    #print(prediction)
    index = str(pred_label[5:])

    return prediction[index][0]
