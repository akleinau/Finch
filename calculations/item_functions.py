import pandas as pd
import panel as pn
import calculations.shap_set_functions as shap_set_functions



class Item:
    def __init__(self, data_loader, data_and_probabilities, index, predict_class, predict_class_label, combined_columns=None):
        self.prediction = get_item_prediction(data_and_probabilities, index)
        self.shap = get_item_shap_values(data_loader, index, predict_class, self.prediction, combined_columns)
        self.data_raw = data_loader.data.iloc[index]
        self.data_prob_raw = data_and_probabilities.iloc[index]
        self.data = get_item_data(self.data_raw)
        self.prob_data =get_item_probability_string(data_and_probabilities, index, self.prediction)
        self.predict_class = predict_class
        self.pred_class_label = predict_class_label
        self.pred_class_str = get_item_class_probability_string(data_and_probabilities, index, predict_class, predict_class_label)
        self.prob_wo_selected_cols = get_prob_wo_selected_cols(data_loader.nn, data_loader.columns, data_loader.means, self.data, self.prediction)
        self.group = 0
        self.scatter_group = 0
        self.scatter_label = 'All'


    def prediction_string(self):
        return pn.pane.Str(self.pred_class_str, sizing_mode="stretch_width", align="center",
                    styles={"font-size": "20px", "text-align": "center"})

    def table(self):
        return self.data


def get_item_shap_values(data_loader, index, predict_class, item_prediction, combined_columns=None):
    item = data_loader.data.iloc[[index]]
    item = item.reset_index(drop=True)
    shap_explanations = shap_set_functions.calc_shap_values(item, data_loader.means, data_loader.nn, data_loader.columns, combined_columns)
    shap_values = pd.DataFrame(shap_explanations.values,
                               columns=shap_explanations.feature_names)
    # pivot the data, so that each row contains the feature and the shap value
    shap_values = shap_values.melt(var_name='feature', value_name='shap_value')
    #add with feature values
    shap_values['feature_label'] = shap_values['feature'].map(lambda x: get_feature_label(x, item))
    shap_values['feature_label_short'] = shap_values['feature_label'].map(lambda x: x[:22] + '...' if len(x) > 25 else x)

    # depending on predict_class, we may have to invert
    if item_prediction != predict_class:
        shap_values['shap_value'] = shap_values['shap_value'] * -1

    # add column containing the absolute value of the shap value
    shap_values['abs_shap_value'] = shap_values['shap_value'].abs()
    shap_values['positive'] = shap_values['shap_value'].map(lambda x: 'pos' if x > 0 else 'neg')
    # sort by the absolute value of the shap value
    combined_item = shap_values.sort_values(by='abs_shap_value', ascending=True)

    return combined_item

def get_feature_label(feature, item):
    feature_split = feature.split(', ')
    if len(feature_split) > 1:
        return ', '.join([get_feature_label(f, item) for f in feature_split])
    else:
        return feature + " = " + str(item[feature].values[0])


def get_item_data(item):
    item = pd.DataFrame({'feature': item.index, 'value': item.values})
    return item


def get_item_probability_string(data, index, prob):
    return "Prediction: " + prob[5:] + "  " + "{:10.0f}".format(data.iloc[index][prob] * 100) + "% certainty"

def get_item_class_probability_string(data, index, predict_class, label):
    return "Probability of " + label + ": " + "{:10.0f}".format(data.iloc[index][predict_class] * 100) + "%"


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
