import pandas as pd
import panel as pn
import calculations.shap_set_functions as shap_set_functions



class Item:
    def __init__(self, data_loader, data_and_probabilities, type, index, custom_content, predict_class, predict_class_label, combined_columns=None):
        self.data_loader = data_loader
        if data_loader.type == 'classification':
            self.prediction = get_item_prediction(data_and_probabilities, index)
        else:
            self.prediction = "prob_Y"
        self.type = type
        if type == 'predefined' or type == 'global':
            self.data_raw = data_loader.data.iloc[[index]]
            self.data_raw = self.data_raw.reset_index(drop=True)
            self.data_prob_raw = data_and_probabilities.iloc[index]
        else:
            self.data_raw = extract_data_from_custom_content(custom_content, data_loader)
            self.data_prob_raw = data_loader.combine_data_and_results(self.data_raw).iloc[0]

        self.data = get_item_data(self.data_raw)
        self.data_series = get_item_Series(self.data_raw)
        self.data_reduced = self.data[~self.data['feature'].str.startswith('truth')]
        #self.shap = get_shap(type, data_loader, self.data_raw, predict_class, self.prediction, combined_columns)
        self.predict_class = predict_class
        self.pred_class_label = predict_class_label
        self.prob_class = self.data_prob_raw[predict_class]
        self.pred_class_str = self.get_item_class_probability_string()
        self.prob_wo_selected_cols = get_prob_wo_selected_cols(data_loader.nn, data_loader.columns, data_loader.means, self.data, self.prediction)
        self.group = 0
        self.scatter_group = 0
        self.scatter_label = 'All'


    def prediction_string(self):
        return pn.pane.Str(self.pred_class_str, sizing_mode="stretch_width", align="center",
                    styles={"font-size": "20px", "text-align": "center"})

    def table(self):
        return self.data

    def get_item_class_probability_string(self):
        if self.type == 'global':
            return ""
        if self.data_loader.type == 'regression':
            return "Prediction: " + "{:.2f}".format(self.prob_class)
        return "Probability of " + self.pred_class_label + ": " + "{:10.0f}".format(self.prob_class * 100) + "%"

def extract_data_from_custom_content(custom_content, data_loader):
    data = {}
    for item in custom_content:
        #check if it is an input widget and not a button
        if hasattr(item, 'value') and not hasattr(item, 'clicks'):
            if (item.value is not None) and (item.value != ''):
                data[item.name] = item.value
            else:
                data[item.name] = data_loader.means[item.name]
    data = pd.DataFrame(data, index=[0])
    return data

def get_item_shap_values(data_loader, item, predict_class, item_prediction, combined_columns=None):
    #print(item)
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

def get_global_shap_values(data_loader, item, predict_class, item_prediction, combined_columns=None):
    #print(item)
    #random subset for efficiency reasons
    data = data_loader.data.sample(n=10, random_state=1)
    shap_explanations = shap_set_functions.calc_shap_values(data, data_loader.means, data_loader.nn, data_loader.columns, combined_columns)
    shap_values = pd.DataFrame(shap_explanations.values,
                               columns=shap_explanations.feature_names)
    # take abs of shap values
    shap_values = shap_values.abs()
    # get the mean of the shap values
    shap_values = shap_values.mean()
    shap_values = pd.DataFrame(shap_values, columns=['shap_value'])
    shap_values['feature'] = shap_values.index
    shap_values.reset_index(drop=True, inplace=True)

    #add labels
    shap_values['feature_label'] = shap_values['feature']
    shap_values['feature_label_short'] = shap_values['feature_label'].map(
        lambda x: x[:22] + '...' if len(x) > 25 else x)

    #absolutes, a bit unnecessary here
    shap_values['abs_shap_value'] = shap_values['shap_value']
    shap_values['positive'] = shap_values['shap_value'].map(lambda x: 'pos')

    # sort by the absolute value of the shap value
    combined_item = shap_values.sort_values(by='abs_shap_value', ascending=True)

    return combined_item


def get_shap(type, data_loader, data_raw, predict_class, prediction, combined_columns=None):
    if type == 'global':

        return get_global_shap_values(data_loader, data_raw, predict_class, prediction, combined_columns)
    else:
        return get_item_shap_values(data_loader, data_raw, predict_class, prediction, combined_columns)

def get_feature_label(feature, item):
    feature_split = feature.split(', ')
    if len(feature_split) > 1:
        return ', '.join([get_feature_label(f, item) for f in feature_split])
    else:
        return feature + " = " + "{:.2f}".format(item[feature].values[0])

def get_item_Series(item):
    return item.iloc[0]

def get_item_data(item):
    item = item.iloc[0]
    item = pd.DataFrame({'feature': item.index, 'value': item.values})
    return item


def get_item_prediction(data, index):
    return data.iloc[index]['prediction']


def get_prob_wo_selected_cols(nn, all_selected_cols, means, item, pred_label):
    item_df = pd.DataFrame(item['value'].values, index=item['feature'].values).T
    new_item = means.copy()

    # replace the values of the selected columns with the mean
    for col in all_selected_cols:
        new_item[col] = item_df[col].iloc[0]

    # calculate the prediction without the selected columns
    predict = nn.predict_proba if hasattr(nn, 'predict_proba') else nn.predict

    prediction = predict(new_item)
    classes = nn.classes_ if hasattr(nn, 'classes_') else ['Y']
    prediction = pd.DataFrame(prediction, columns=[str(a) for a in classes])
    #print(prediction)
    index = str(pred_label[5:])

    return prediction[index][0]
