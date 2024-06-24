import pandas as pd


def get_similar_items(data, item, col_white_list):
    use_shap = False
    # standardize the data
    data_std = data.copy()
    item_data = item.data_raw.copy()

    if len(col_white_list) == 0:
        columns = list(data.columns)
        excluded_columns = ['prob_', 'scatter', 'prediction', 'group', 'truth']

        columns = [col for col in columns if not any([excluded in col for excluded in excluded_columns])]
        item_columns = [col for col in item_data.columns if not any([excluded in col for excluded in excluded_columns])]
        columns = [col for col in columns if col in item_columns]
    else:
        columns = col_white_list

    for col in columns:
        mean = data_std[col].mean()
        std = data_std[col].std()
        data_std[col] = (data_std[col] - mean) / std
        item_data[col] = (item_data[col] - mean) / std

    # calculate distance to item, using shap as weights

    if use_shap:
        shap_widened = {}
        for row in item.shap.iterrows():
            splits = row[1]['feature'].split(', ')
            for split in splits:
                shap_widened[split] = 0.1 + row[1]['abs_shap_value']
        shap_widened = pd.DataFrame(shap_widened, index=[0])

        data_std['distance'] = 0
        for col in columns:
            data_std['distance'] += shap_widened[col][0]*(data_std[col] - item_data[col][0])**2
    else:
        data_std['distance'] = 0
        for col in columns:
            data_std['distance'] += (data_std[col] - item_data[col][0])**2

    # get the 10% closest items
    data_std = data_std.sort_values(by='distance')
    num_items = min(max(int(len(data) * 0.05), 30), len(data_std))
    closest_density = data_std.head(num_items)
    min_distance = len(columns) * 0.1
    closest_distance = data_std[data_std['distance'] <= min_distance]
    combined_indexes = closest_density.index.union(closest_distance.index)

    #map back to original data
    data = data[data.index.isin(combined_indexes)]

    return data