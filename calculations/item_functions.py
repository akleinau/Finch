import pandas as pd
import panel as pn
from panel.viewable import Viewer
import param

class Item(Viewer):
    data_reduced = param.ClassSelector(class_=pd.DataFrame)

    def __init__(self, data_loader, data_and_probabilities, type, index, custom_content, predict_class,
                 predict_class_label, combined_columns=None, **params):
        super().__init__(**params)
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
            self.data_prob_raw = data_loader.combine_data_and_results(data=self.data_raw).iloc[0]

        self.data = get_item_data(self.data_raw)
        self.data_series = get_item_Series(self.data_raw)
        self.data_reduced = self.data[~self.data['feature'].str.startswith('truth')]
        #self.shap = get_shap(type, data_loader, self.data_raw, predict_class, self.prediction, combined_columns)
        self.predict_class = predict_class
        self.pred_class_label = predict_class_label
        self.prob_class = self.data_prob_raw[predict_class]
        self.pred_class_str = self.get_item_class_probability_string()
        self.prob_only_selected_cols = get_prob_only_selected_cols(data_loader.nn, data_loader.columns, data_loader.means, self.data, self.prediction)
        self.group = 0
        self.scatter_group = 0
        self.scatter_label = 'All'

    @param.depends('data_reduced')
    def __panel__(self):
        return self.data_reduced


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


def get_prob_only_selected_cols(nn, all_selected_cols, means, item, pred_label):
    item_df = pd.DataFrame(item['value'].values, index=item['feature'].values).T
    new_item = means.copy()

    # replace the values of the selected columns with the the item values, rest stays at mean
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
