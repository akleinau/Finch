import pandas as pd


def get_similar_items(data, item, exclude_cols):
    # standardize the data
    data_std = data.copy()
    item_data = item.data_raw.copy()
    columns = list(data.columns)
    excluded_columns = ['prob_', 'scatter', 'prediction', 'group']
    if len(exclude_cols) > 0:
        excluded_columns.extend(exclude_cols)
    columns = [col for col in columns if not any([excluded in col for excluded in excluded_columns])]

    for col in columns:
        mean = data_std[col].mean()
        std = data_std[col].std()
        data_std[col] = (data_std[col] - mean) / std
        item_data[col] = (item_data[col] - mean) / std

    # calculate distance to item, using shap as weights

    shap_widened = pd.DataFrame()
    for row in item.shap.iterrows():
        splits = row[1]['feature'].split(', ')
        for split in splits:
            shap_widened[split] = row[1]['shap_value']
    data_std['distance'] = (shap_widened[columns]*(data[columns] - item_data[columns])**2).sum(axis=1)

    # get the 10% closest items
    data_std = data_std.sort_values(by='distance')
    data_std = data_std.head(int(len(data) * 0.1))

    #map back to original data
    data = data[data.index.isin(data_std.index)]

    return data